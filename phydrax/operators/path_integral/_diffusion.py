#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import cast, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ...domain._function import DomainFunction
from ._discretization import PathDiscretization
from ._potential import _as_point_time_callable, PotentialLike


DriftLike: TypeAlias = PotentialLike | None
DiffusionLike: TypeAlias = ArrayLike | PotentialLike


def _initial_state(x0: ArrayLike, /) -> Array:
    state = jnp.asarray(x0, dtype=float)
    if state.ndim == 0:
        state = state[None]
    if int(state.shape[-1]) < 1:
        raise ValueError("x0 state dimension must be non-empty.")
    return eqx.error_if(state, ~jnp.all(jnp.isfinite(state)), "x0 must be finite.")


def _evaluate_drift(
    drift: DriftLike,
    state: Array,
    time: Array,
    /,
    *,
    position_var: str,
    time_var: str,
    key: Key[Array, ""],
) -> Array:
    if drift is None:
        return jnp.zeros_like(state)
    state_dim = int(state.shape[-1])
    flat_state = jnp.reshape(state, (-1, state_dim))
    drift_fn = _as_point_time_callable(
        drift,
        position_var=position_var,
        time_var=time_var,
        key=key,
        role="Drift",
    )
    values = jnp.asarray(jax.vmap(drift_fn, in_axes=(0, None))(flat_state, time))
    if state_dim == 1 and values.shape == (flat_state.shape[0],):
        values = values[:, None]
    if values.shape != flat_state.shape:
        raise ValueError(
            "drift must return one state vector per point; "
            f"got {values.shape}, expected {flat_state.shape}."
        )
    values = eqx.error_if(
        values,
        ~jnp.all(jnp.isfinite(values)),
        "drift values must be finite.",
    )
    return jnp.reshape(values, state.shape)


def _apply_diffusion(
    diffusion: DiffusionLike,
    state: Array,
    time: Array,
    normal_increment: Array,
    /,
    *,
    position_var: str,
    time_var: str,
    key: Key[Array, ""],
) -> Array:
    state_dim = int(state.shape[-1])
    flat_state = jnp.reshape(state, (-1, state_dim))
    flat_increment = jnp.reshape(normal_increment, (-1, state_dim))

    if callable(diffusion) or isinstance(diffusion, DomainFunction):
        diffusion_fn = _as_point_time_callable(
            cast(PotentialLike, diffusion),
            position_var=position_var,
            time_var=time_var,
            key=key,
            role="Diffusion",
        )
        coefficient = jnp.asarray(
            jax.vmap(diffusion_fn, in_axes=(0, None))(flat_state, time)
        )
        if coefficient.shape == (flat_state.shape[0],):
            out = coefficient[:, None] * flat_increment
        elif coefficient.shape == flat_state.shape:
            out = coefficient * flat_increment
        elif coefficient.shape == (
            flat_state.shape[0],
            state_dim,
            state_dim,
        ):
            out = oe.contract("nij,nj->ni", coefficient, flat_increment)
        else:
            raise ValueError(
                "callable diffusion must return a scalar, diagonal vector, or "
                f"square matrix per point; got shape {coefficient.shape}."
            )
    else:
        coefficient = jnp.asarray(diffusion, dtype=float)
        if coefficient.ndim == 0:
            out = coefficient * flat_increment
        elif coefficient.shape == (state_dim,):
            out = coefficient * flat_increment
        elif coefficient.shape == (state_dim, state_dim):
            out = oe.contract("ij,nj->ni", coefficient, flat_increment)
        else:
            raise ValueError(
                "diffusion must be a scalar, state_dim vector, state_dim square "
                f"matrix, or compatible callable; got shape {coefficient.shape}."
            )
    out = eqx.error_if(
        out,
        ~jnp.all(jnp.isfinite(out)),
        "diffusion increments must be finite.",
    )
    return jnp.reshape(out, state.shape)


def diffusion_paths_from_noise(
    drift: DriftLike,
    diffusion: DiffusionLike,
    x0: ArrayLike,
    noise: ArrayLike,
    /,
    *,
    slicing: PathDiscretization,
    position_var: str = "x",
    time_var: str = "t",
    key: Key[Array, ""] = DOC_KEY0,
) -> Array:
    r"""Simulate Itô diffusion paths from explicit Euler--Maruyama noise.

    ``noise`` has shape ``(..., num_paths, num_steps, state_dim)``. The returned
    paths have the same leading axes and a node axis of length ``num_steps + 1``.
    """
    start = _initial_state(x0)
    z = jnp.asarray(noise, dtype=float)
    if z.ndim < 3:
        raise ValueError("noise must have shape (..., num_paths, num_steps, state_dim).")
    if int(z.shape[-2]) != slicing.num_steps:
        raise ValueError(
            "noise step axis must match slicing.num_steps; "
            f"got {int(z.shape[-2])} and {slicing.num_steps}."
        )
    if int(z.shape[-1]) != int(start.shape[-1]):
        raise ValueError("noise state dimension must match x0.")
    batch_shape = jnp.broadcast_shapes(start.shape[:-1], z.shape[:-3])
    start = jnp.broadcast_to(start, batch_shape + (int(start.shape[-1]),))
    z = jnp.broadcast_to(
        z,
        batch_shape + (int(z.shape[-3]), slicing.num_steps, int(z.shape[-1])),
    )
    z = eqx.error_if(z, ~jnp.all(jnp.isfinite(z)), "noise must be finite.")
    num_paths = int(z.shape[-3])
    state = jnp.broadcast_to(
        start[..., None, :],
        batch_shape + (num_paths, int(start.shape[-1])),
    )
    step_noise = jnp.moveaxis(z, -2, 0)
    step_keys = jr.split(key, slicing.num_steps)

    def step(current, xs):
        time, normal, step_key = xs
        drift_key, diffusion_key = jr.split(step_key)
        drift_value = _evaluate_drift(
            drift,
            current,
            time,
            position_var=position_var,
            time_var=time_var,
            key=drift_key,
        )
        stochastic = _apply_diffusion(
            diffusion,
            current,
            time,
            normal,
            position_var=position_var,
            time_var=time_var,
            key=diffusion_key,
        )
        next_state = (
            current + slicing.dt * drift_value + jnp.sqrt(slicing.dt) * stochastic
        )
        return next_state, next_state

    _, history = jax.lax.scan(
        step,
        state,
        (slicing.times[:-1], step_noise, step_keys),
    )
    history = jnp.moveaxis(history, 0, -2)
    return jnp.concatenate((state[..., None, :], history), axis=-2)


def sample_diffusion_paths(
    drift: DriftLike,
    diffusion: DiffusionLike,
    x0: ArrayLike,
    /,
    *,
    slicing: PathDiscretization,
    num_paths: int,
    position_var: str = "x",
    time_var: str = "t",
    key: Key[Array, ""] = DOC_KEY0,
) -> Array:
    """Draw Euler--Maruyama paths for a finite-dimensional Itô diffusion."""
    count = int(num_paths)
    if count < 1:
        raise ValueError("num_paths must be at least one.")
    start = _initial_state(x0)
    noise_key, dynamics_key = jr.split(key)
    noise = jr.normal(
        noise_key,
        start.shape[:-1] + (count, slicing.num_steps, int(start.shape[-1])),
        dtype=start.dtype,
    )
    return diffusion_paths_from_noise(
        drift,
        diffusion,
        start,
        noise,
        slicing=slicing,
        position_var=position_var,
        time_var=time_var,
        key=dynamics_key,
    )


__all__ = [
    "DiffusionLike",
    "DriftLike",
    "diffusion_paths_from_noise",
    "sample_diffusion_paths",
]
