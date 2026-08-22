#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ...discretization import TemporalMesh
from ._potential import _as_potential_callable, PotentialLike
from ._sampling import _positive_scalar


def _paths_array(paths: ArrayLike, slicing: TemporalMesh, /) -> Array:
    out = jnp.asarray(paths, dtype=float)
    if out.ndim < 2:
        raise ValueError("paths must have shape (..., num_nodes, state_dim).")
    if int(out.shape[-2]) != slicing.num_nodes:
        raise ValueError(
            "paths node axis must match slicing.num_nodes; "
            f"got {int(out.shape[-2])} and {slicing.num_nodes}."
        )
    if int(out.shape[-1]) < 1:
        raise ValueError("paths state dimension must be non-empty.")
    return eqx.error_if(out, ~jnp.all(jnp.isfinite(out)), "paths must be finite.")


def kinetic_action(
    paths: ArrayLike,
    /,
    *,
    slicing: TemporalMesh,
    mass: ArrayLike,
) -> Array:
    r"""Evaluate the time-sliced Euclidean kinetic action.

    For path nodes $q_k$, this returns
    $\sum_k m\lVert q_{k+1}-q_k\rVert^2/(2\,\Delta t)$.
    """
    q = _paths_array(paths, slicing)
    mass_arr = _positive_scalar("mass", mass)
    increments = q[..., 1:, :] - q[..., :-1, :]
    squared = jnp.sum(increments * increments, axis=-1)
    return 0.5 * mass_arr * jnp.sum(squared, axis=-1) / slicing.dt


def potential_action(
    paths: ArrayLike,
    potential: PotentialLike,
    /,
    *,
    slicing: TemporalMesh,
    position_var: str = "q",
    time_var: str = "t",
    key: Key[Array, ""] = DOC_KEY0,
) -> Array:
    r"""Evaluate a real scalar potential along path-segment midpoints."""
    potential_fn = _as_potential_callable(
        potential,
        position_var=position_var,
        time_var=time_var,
        key=key,
    )
    q = _paths_array(paths, slicing)
    mid_q = 0.5 * (q[..., :-1, :] + q[..., 1:, :])
    leading_shape = mid_q.shape[:-2]
    flat_q = jnp.reshape(mid_q, (-1, int(mid_q.shape[-1])))
    times = jnp.broadcast_to(
        slicing.midpoints,
        leading_shape + (slicing.num_steps,),
    )
    flat_t = jnp.reshape(times, (-1,))
    values = jax.vmap(potential_fn)(flat_q, flat_t)
    values = jnp.asarray(values)
    if values.shape != flat_t.shape:
        raise ValueError(
            "potential must return one scalar per (q, t) point; "
            f"got output shape {values.shape} for input shape {flat_t.shape}."
        )
    if jnp.iscomplexobj(values):
        raise TypeError("Euclidean potential values must be real.")
    values = eqx.error_if(
        values,
        ~jnp.all(jnp.isfinite(values)),
        "Euclidean potential values must be finite.",
    )
    values = jnp.reshape(values, leading_shape + (slicing.num_steps,))
    return slicing.dt * jnp.sum(values, axis=-1)


def discrete_euclidean_action(
    paths: ArrayLike,
    potential: PotentialLike,
    /,
    *,
    slicing: TemporalMesh,
    mass: ArrayLike,
    position_var: str = "q",
    time_var: str = "t",
    key: Key[Array, ""] = DOC_KEY0,
) -> Array:
    r"""Evaluate midpoint-discretized kinetic plus potential Euclidean action."""
    return kinetic_action(paths, slicing=slicing, mass=mass) + potential_action(
        paths,
        potential,
        slicing=slicing,
        position_var=position_var,
        time_var=time_var,
        key=key,
    )


__all__ = [
    "discrete_euclidean_action",
    "kinetic_action",
    "potential_action",
]
