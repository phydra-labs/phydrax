#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ._action import potential_action
from ._discretization import PathDiscretization
from ._estimate import (
    _estimate_positive_log_sums,
    _estimate_positive_log_weights,
    PathIntegralEstimate,
)
from ._potential import PotentialLike
from ._sampling import (
    _endpoints,
    _positive_scalar,
    brownian_bridge_from_noise,
)


def free_euclidean_kernel(
    x0: ArrayLike,
    x1: ArrayLike,
    /,
    *,
    duration: ArrayLike,
    mass: ArrayLike = 1.0,
    hbar: ArrayLike = 1.0,
) -> Array:
    r"""Analytic free-particle Euclidean kernel in finite-dimensional space."""
    start, end = _endpoints(x0, x1)
    duration_arr = _positive_scalar("duration", duration)
    mass_arr = _positive_scalar("mass", mass)
    hbar_arr = _positive_scalar("hbar", hbar)
    dimension = int(start.shape[-1])
    displacement_sq = jnp.sum((end - start) ** 2, axis=-1)
    normalization = jnp.power(
        mass_arr / (2.0 * jnp.pi * hbar_arr * duration_arr),
        0.5 * dimension,
    )
    return normalization * jnp.exp(
        -mass_arr * displacement_sq / (2.0 * hbar_arr * duration_arr)
    )


def _free_estimate(scale: Array, count: int, /) -> PathIntegralEstimate:
    return PathIntegralEstimate(
        value=scale,
        standard_error=jnp.zeros_like(scale),
        effective_sample_size=jnp.full_like(scale, float(count)),
        log_mean_weight=jnp.zeros_like(scale),
        num_paths=int(count),
    )


def _log_weights(
    paths: Array,
    potential: PotentialLike,
    /,
    *,
    slicing: PathDiscretization,
    hbar: Array,
    position_var: str,
    time_var: str,
    key: Key[Array, ""],
) -> Array:
    action = potential_action(
        paths,
        potential,
        slicing=slicing,
        position_var=position_var,
        time_var=time_var,
        key=key,
    )
    return -action / hbar


def euclidean_kernel_from_noise(
    potential: PotentialLike | None,
    x0: ArrayLike,
    x1: ArrayLike,
    noise: ArrayLike,
    /,
    *,
    slicing: PathDiscretization,
    mass: ArrayLike = 1.0,
    hbar: ArrayLike = 1.0,
    position_var: str = "q",
    time_var: str = "t",
    key: Key[Array, ""] = DOC_KEY0,
) -> PathIntegralEstimate:
    r"""Estimate a Euclidean kernel from explicit standard-normal bridge noise."""
    mass_arr = _positive_scalar("mass", mass)
    hbar_arr = _positive_scalar("hbar", hbar)
    start, end = _endpoints(x0, x1)
    z = jnp.asarray(noise, dtype=float)
    if z.ndim < 3:
        raise ValueError("noise must have shape (..., num_paths, num_steps, state_dim).")
    count = int(z.shape[-3])
    if count < 1:
        raise ValueError("noise must contain at least one path.")
    scale = free_euclidean_kernel(
        start,
        end,
        duration=slicing.duration,
        mass=mass_arr,
        hbar=hbar_arr,
    )
    if potential is None:
        return _free_estimate(scale, count)

    paths = brownian_bridge_from_noise(
        z,
        start,
        end,
        slicing=slicing,
        diffusion=hbar_arr / mass_arr,
    )
    log_weights = _log_weights(
        paths,
        potential,
        slicing=slicing,
        hbar=hbar_arr,
        position_var=position_var,
        time_var=time_var,
        key=key,
    )
    return _estimate_positive_log_weights(log_weights, scale=scale)


def euclidean_kernel(
    potential: PotentialLike | None,
    x0: ArrayLike,
    x1: ArrayLike,
    /,
    *,
    slicing: PathDiscretization,
    mass: ArrayLike = 1.0,
    hbar: ArrayLike = 1.0,
    num_paths: int,
    chunk_size: int | None = None,
    position_var: str = "q",
    time_var: str = "t",
    key: Key[Array, ""] = DOC_KEY0,
) -> PathIntegralEstimate:
    r"""Estimate a fixed-endpoint Euclidean propagator with Brownian bridges.

    The free kinetic measure is sampled exactly. Only the midpoint-discretized
    potential action is reweighted, which avoids bounded path-coordinate truncation.
    """
    mass_arr = _positive_scalar("mass", mass)
    hbar_arr = _positive_scalar("hbar", hbar)
    start, end = _endpoints(x0, x1)
    count = int(num_paths)
    if count < 1:
        raise ValueError("num_paths must be at least one.")
    scale = free_euclidean_kernel(
        start,
        end,
        duration=slicing.duration,
        mass=mass_arr,
        hbar=hbar_arr,
    )
    if potential is None:
        return _free_estimate(scale, count)

    chunk = min(count, 1024) if chunk_size is None else int(chunk_size)
    if chunk < 1:
        raise ValueError("chunk_size must be at least one.")
    chunk = min(chunk, count)
    num_chunks = (count + chunk - 1) // chunk
    indices = jnp.arange(num_chunks, dtype=jnp.int32)
    batch_shape = start.shape[:-1]
    state_dim = int(start.shape[-1])
    single_path_noise_shape = batch_shape + (slicing.num_steps, state_dim)
    path_offsets = jnp.arange(chunk, dtype=jnp.int32)
    potential_key = jr.fold_in(key, count)
    initial = (
        jnp.full(batch_shape, -jnp.inf, dtype=float),
        jnp.full(batch_shape, -jnp.inf, dtype=float),
    )

    def accumulate(carry, index):
        log_sum, log_sum_sq = carry
        path_indices = index * chunk + path_offsets
        path_keys = jax.vmap(lambda path_index: jr.fold_in(key, path_index))(path_indices)
        noise = jax.vmap(
            lambda path_key: jr.normal(
                path_key,
                single_path_noise_shape,
                dtype=start.dtype,
            )
        )(path_keys)
        noise = jnp.moveaxis(noise, 0, len(batch_shape))
        paths = brownian_bridge_from_noise(
            noise,
            start,
            end,
            slicing=slicing,
            diffusion=hbar_arr / mass_arr,
        )
        log_weights = _log_weights(
            paths,
            potential,
            slicing=slicing,
            hbar=hbar_arr,
            position_var=position_var,
            time_var=time_var,
            key=potential_key,
        )
        valid_count = jnp.minimum(chunk, count - index * chunk)
        valid = path_offsets < valid_count
        valid = jnp.reshape(valid, (1,) * len(batch_shape) + (chunk,))
        log_weights = jnp.where(valid, log_weights, -jnp.inf)
        chunk_log_sum = jsp.logsumexp(log_weights, axis=-1)
        chunk_log_sum_sq = jsp.logsumexp(2.0 * log_weights, axis=-1)
        return (
            jnp.logaddexp(log_sum, chunk_log_sum),
            jnp.logaddexp(log_sum_sq, chunk_log_sum_sq),
        ), None

    (log_sum, log_sum_sq), _ = jax.lax.scan(
        accumulate,
        initial,
        indices,
    )
    return _estimate_positive_log_sums(
        log_sum,
        log_sum_sq,
        count=count,
        scale=scale,
    )


__all__ = [
    "euclidean_kernel",
    "euclidean_kernel_from_noise",
    "free_euclidean_kernel",
]
