#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ._discretization import PathDiscretization


def _endpoints(x0: ArrayLike, x1: ArrayLike, /) -> tuple[Array, Array]:
    start = jnp.asarray(x0, dtype=float)
    end = jnp.asarray(x1, dtype=float)
    if start.ndim == 0:
        start = start[None]
    if end.ndim == 0:
        end = end[None]
    if int(start.shape[-1]) != int(end.shape[-1]):
        raise ValueError(
            "Path endpoints must have matching state dimensions; "
            f"got {start.shape} and {end.shape}."
        )
    try:
        batch_shape = jnp.broadcast_shapes(start.shape[:-1], end.shape[:-1])
    except ValueError as error:
        raise ValueError(
            "Path endpoint batch dimensions must be broadcast-compatible; "
            f"got {start.shape} and {end.shape}."
        ) from error
    state_dim = int(start.shape[-1])
    start = jnp.broadcast_to(start, batch_shape + (state_dim,))
    end = jnp.broadcast_to(end, batch_shape + (state_dim,))
    if int(start.shape[-1]) < 1:
        raise ValueError("Path endpoint state dimension must be non-empty.")
    start = eqx.error_if(start, ~jnp.all(jnp.isfinite(start)), "x0 must be finite.")
    end = eqx.error_if(end, ~jnp.all(jnp.isfinite(end)), "x1 must be finite.")
    return start, end


def _positive_scalar(name: str, value: ArrayLike, /) -> Array:
    out = jnp.asarray(value, dtype=float)
    if out.shape != ():
        raise ValueError(f"{name} must be scalar, got shape {out.shape}.")
    return eqx.error_if(
        out,
        (~jnp.isfinite(out)) | (out <= 0.0),
        f"{name} must be finite and positive.",
    )


def brownian_bridge_from_noise(
    noise: ArrayLike,
    x0: ArrayLike,
    x1: ArrayLike,
    /,
    *,
    slicing: PathDiscretization,
    diffusion: ArrayLike = 1.0,
) -> Array:
    r"""Construct endpoint-conditioned Brownian paths from standard-normal increments.

    `noise` must have shape
    `endpoint_batch + (num_paths, slicing.num_steps, state_dim)`. The returned
    paths have the same leading shape and `slicing.num_steps + 1` path nodes.
    """
    start, end = _endpoints(x0, x1)
    z = jnp.asarray(noise, dtype=float)
    if z.ndim < 3:
        raise ValueError("noise must have shape (..., num_paths, num_steps, state_dim).")
    if int(z.shape[-2]) != slicing.num_steps:
        raise ValueError(
            "noise step axis must match slicing.num_steps; "
            f"got {int(z.shape[-2])} and {slicing.num_steps}."
        )
    if int(z.shape[-1]) != int(start.shape[-1]):
        raise ValueError(
            "noise state dimension must match the endpoints; "
            f"got {int(z.shape[-1])} and {int(start.shape[-1])}."
        )
    if z.shape[:-3] != start.shape[:-1]:
        raise ValueError(
            "noise endpoint batch dimensions must match x0/x1; "
            f"got {z.shape[:-3]} and {start.shape[:-1]}."
        )
    if int(z.shape[-3]) < 1:
        raise ValueError("noise must contain at least one path.")

    diffusion_arr = _positive_scalar("diffusion", diffusion)
    increments = jnp.sqrt(diffusion_arr * slicing.dt) * z
    zero = jnp.zeros(z.shape[:-2] + (1, z.shape[-1]), dtype=z.dtype)
    motion = jnp.concatenate((zero, jnp.cumsum(increments, axis=-2)), axis=-2)

    endpoint_batch_ndim = start.ndim - 1
    s = (slicing.times - slicing.t0) / slicing.duration
    s_shape = (1,) * endpoint_batch_ndim + (1, slicing.num_nodes, 1)
    s = jnp.reshape(s, s_shape)

    start_b = jnp.reshape(start, start.shape[:-1] + (1, 1, int(start.shape[-1])))
    end_b = jnp.reshape(end, end.shape[:-1] + (1, 1, int(end.shape[-1])))
    linear = start_b + s * (end_b - start_b)
    bridge_fluctuation = motion - s * motion[..., -1:, :]
    paths = linear + bridge_fluctuation

    # Make endpoint exactness an algebraic invariant, not a tolerance-based property.
    paths = paths.at[..., 0, :].set(start_b[..., 0, :])
    paths = paths.at[..., -1, :].set(end_b[..., 0, :])
    return paths


def sample_brownian_bridge(
    x0: ArrayLike,
    x1: ArrayLike,
    /,
    *,
    slicing: PathDiscretization,
    num_paths: int,
    diffusion: ArrayLike = 1.0,
    key: Key[Array, ""] = DOC_KEY0,
) -> Array:
    """Sample fixed-endpoint Brownian paths on a uniform time slicing."""
    start, end = _endpoints(x0, x1)
    count = int(num_paths)
    if count < 1:
        raise ValueError("num_paths must be at least one.")
    shape = start.shape[:-1] + (
        count,
        slicing.num_steps,
        int(start.shape[-1]),
    )
    noise = jr.normal(key, shape, dtype=start.dtype)
    return brownian_bridge_from_noise(
        noise,
        start,
        end,
        slicing=slicing,
        diffusion=diffusion,
    )


__all__ = [
    "brownian_bridge_from_noise",
    "sample_brownian_bridge",
]
