#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._probability import AbstractProbabilityLaw
from .._strict import StrictModule
from ..domain._measure import MeasureKind
from ..metrix import AbstractRiemannianManifold


class ManifoldProbabilityLaw(AbstractProbabilityLaw):
    """Normalized law with density relative to one manifold's Riemannian volume."""

    manifold: AbstractRiemannianManifold
    sampler: Any
    log_density: Any
    law_id: str = eqx.field(static=True)

    def __init__(self, manifold, sampler, log_density, /, *, law_id: str):
        if not isinstance(manifold, AbstractRiemannianManifold):
            raise TypeError("manifold must implement AbstractRiemannianManifold.")
        if not callable(sampler) or not callable(log_density):
            raise TypeError("sampler and log_density must be callable.")
        if not law_id:
            raise ValueError("law_id must be non-empty.")
        self.manifold = manifold
        self.sampler = sampler
        self.log_density = log_density
        self.law_id = law_id

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.manifold.point_shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def density_measure_kind(self) -> MeasureKind:
        return "riemannian"

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        value = jnp.asarray(self.sampler(key, tuple(sample_shape)))
        if value.shape != tuple(sample_shape) + self.event_shape:
            raise ValueError("Manifold sampler returned an incompatible shape.")
        return value

    def contains(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        rank = len(self.event_shape)
        if array.ndim < rank or tuple(array.shape[-rank:]) != self.event_shape:
            raise ValueError("Manifold value must end in the manifold point shape.")
        rank = len(self.event_shape)
        leading = array.shape[:-rank] if rank else array.shape
        flat = array.reshape((-1,) + self.event_shape)
        result = jax.vmap(self.manifold.contains)(flat)
        return result.reshape(leading)

    def log_prob(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        rank = len(self.event_shape)
        if array.ndim < rank or tuple(array.shape[-rank:]) != self.event_shape:
            raise ValueError("Manifold value must end in the manifold point shape.")
        rank = len(self.event_shape)
        leading = array.shape[:-rank] if rank else array.shape
        flat = array.reshape((-1,) + self.event_shape)
        density = jax.vmap(self.log_density)(flat).reshape(leading)
        return jnp.where(self.contains(array), density, -jnp.inf)


class RiemannianScoreField(StrictModule):
    """Tangent score callable validated against one Riemannian manifold."""

    manifold: AbstractRiemannianManifold
    function: Any
    score_id: str = eqx.field(static=True)

    def __init__(self, manifold, function, /, *, score_id: str):
        if not isinstance(manifold, AbstractRiemannianManifold) or not callable(function):
            raise TypeError("Riemannian score requires a manifold and callable.")
        if not score_id:
            raise ValueError("score_id must be non-empty.")
        self.manifold = manifold
        self.function = function
        self.score_id = score_id

    def __call__(self, point: ArrayLike, time: ArrayLike, /, *, key=None) -> Array:
        value = jnp.asarray(point)
        if value.shape != self.manifold.point_shape:
            raise ValueError("Riemannian score evaluation requires one manifold point.")
        if jnp.asarray(time).shape != ():
            raise ValueError("Riemannian score evaluation requires scalar time.")
        value = eqx.error_if(
            value,
            ~self.manifold.contains(value),
            "Riemannian score point lies outside the manifold.",
        )
        score = jnp.asarray(self.function(value, jnp.asarray(time), key=key))
        if score.shape != value.shape:
            raise ValueError("Riemannian score must preserve the manifold point shape.")
        tangent = self.manifold.project_tangent(value, score)
        defect = jnp.linalg.vector_norm(score - tangent)
        return eqx.error_if(
            tangent,
            defect > 1e-7 * (1.0 + jnp.linalg.vector_norm(score)),
            "Riemannian score is not tangent at the supplied point.",
        )


class IsotropicRiemannianDiffusion(StrictModule):
    """Intrinsic scalar-rate diffusion with tangent drift and reverse-score correction."""

    manifold: AbstractRiemannianManifold
    drift_function: Any
    diffusion_rate: Any
    tangent_noise: Any
    terminal_time: float = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        manifold,
        drift,
        diffusion_rate,
        tangent_noise,
        /,
        *,
        terminal_time: float = 1.0,
        process_id: str | None = None,
    ):
        if not isinstance(manifold, AbstractRiemannianManifold):
            raise TypeError("manifold must implement AbstractRiemannianManifold.")
        if not callable(drift) or not callable(diffusion_rate) or not callable(tangent_noise):
            raise TypeError("drift, diffusion_rate, and tangent_noise must be callable.")
        horizon = float(terminal_time)
        if not isfinite(horizon) or horizon <= 0.0:
            raise ValueError("terminal_time must be finite and positive.")
        identifier = process_id or canonical_fingerprint(
            {
                "kind": "isotropic-riemannian-diffusion",
                "manifold_id": manifold.manifold_id,
                "terminal_time": horizon,
            }
        )
        self.manifold = manifold
        self.drift_function = drift
        self.diffusion_rate = diffusion_rate
        self.tangent_noise = tangent_noise
        self.terminal_time = horizon
        self.process_id = identifier

    def drift(self, time, point, /):
        value = jnp.asarray(point)
        drift = jnp.asarray(self.drift_function(jnp.asarray(time), value))
        return self.manifold.project_tangent(value, drift)

    def rate(self, time, /) -> Array:
        value = jnp.asarray(self.diffusion_rate(jnp.asarray(time)), dtype=float).reshape(())
        return eqx.error_if(value, ~jnp.isfinite(value) | (value < 0.0), "Invalid diffusion rate.")

    def reverse_drift(self, reverse_time, point, score, /):
        time = self.terminal_time - jnp.asarray(reverse_time)
        value = jnp.asarray(point)
        tangent_score = self.manifold.project_tangent(value, score)
        return -self.drift(time, value) + self.rate(time) ** 2 * tangent_score

    def probability_flow_drift(self, reverse_time, point, score, /):
        time = self.terminal_time - jnp.asarray(reverse_time)
        value = jnp.asarray(point)
        tangent_score = self.manifold.project_tangent(value, score)
        return -self.drift(time, value) + 0.5 * self.rate(time) ** 2 * tangent_score


class ManifoldDiffusionSample(StrictModule):
    final_state: Array
    trajectory: Array
    valid: Array
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)


def sample_manifold_reverse_diffusion(
    process: IsotropicRiemannianDiffusion,
    score: RiemannianScoreField,
    terminal_state: ArrayLike,
    key: Key[Array, ""],
    /,
    *,
    num_steps: int,
) -> ManifoldDiffusionSample:
    """Fixed-step retraction Euler sampler for intrinsic reverse diffusion."""
    if not isinstance(process, IsotropicRiemannianDiffusion) or not isinstance(
        score, RiemannianScoreField
    ):
        raise TypeError("Manifold reverse sampling requires process and score objects.")
    if process.manifold.manifold_id != score.manifold.manifold_id:
        raise ValueError("Process and score manifolds differ.")
    count = int(num_steps)
    if count <= 0:
        raise ValueError("num_steps must be positive.")
    initial = jnp.asarray(terminal_state)
    initial = eqx.error_if(
        initial,
        ~process.manifold.contains(initial),
        "terminal_state lies outside the manifold.",
    )
    step_size = process.terminal_time / count
    reverse_times = jnp.arange(count, dtype=float) * step_size

    def step(carry, reverse_time):
        point, current_key = carry
        current_key, score_key, noise_key = jr.split(current_key, 3)
        forward_time = process.terminal_time - reverse_time
        score_value = score(point, forward_time, key=score_key)
        drift = process.reverse_drift(reverse_time, point, score_value)
        noise = jnp.asarray(process.tangent_noise(noise_key, point))
        if noise.shape != point.shape:
            raise ValueError("Tangent noise must match the manifold point shape.")
        noise = process.manifold.project_tangent(point, noise)
        tangent_step = step_size * drift + jnp.sqrt(step_size) * process.rate(forward_time) * noise
        destination = process.manifold.retract(point, tangent_step)
        return (destination, current_key), destination

    (final, _), trajectory = jax.lax.scan(step, (initial, key), reverse_times)
    valid = process.manifold.contains(final) & jnp.all(jnp.isfinite(final))
    return ManifoldDiffusionSample(
        final,
        trajectory,
        valid,
        process.process_id,
        "fixed-retraction-euler",
    )


__all__ = [
    "IsotropicRiemannianDiffusion",
    "ManifoldDiffusionSample",
    "ManifoldProbabilityLaw",
    "RiemannianScoreField",
    "sample_manifold_reverse_diffusion",
]
