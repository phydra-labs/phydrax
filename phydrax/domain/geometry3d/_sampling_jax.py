#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Key

from ..._doc import DOC_KEY0


_SDF_INSIDE_TOL = -1e-8


def _sample_unit_cube(
    key: Key[Array, ""],
    num_points: int,
    *,
    dim: int,
    sampler: str,
) -> Array:
    n = int(num_points)
    d = int(dim)
    sampler_ = str(sampler).lower()
    if n == 0:
        return jnp.empty((0, d), dtype=float)
    if d <= 0:
        raise ValueError(f"dim must be positive, got {d}.")

    if sampler_ == "uniform":
        return jr.uniform(key, shape=(n, d), dtype=float)

    if sampler_ != "latin_hypercube":
        raise ValueError("Only sampler='uniform' or 'latin_hypercube' is supported.")

    key_perm, key_u = jr.split(key, 2)
    perm_keys = jr.split(key_perm, d)
    u_keys = jr.split(key_u, d)
    n_f = jnp.asarray(n, dtype=float)

    def _lhs_1d(kp, ku):
        perm = jr.permutation(kp, n)
        u = jr.uniform(ku, shape=(n,), dtype=float)
        return (perm.astype(float) + u) / n_f

    coords = jax.vmap(_lhs_1d)(perm_keys, u_keys)  # (d, n)
    return coords.T


def sample_interior_rejection_jax(
    adf: Callable[[Array], Array],
    bounds: Array,
    num_points: int,
    *,
    where: Callable[[Array], Bool[Array, ""]] | None = None,
    sampler: str = "latin_hypercube",
    key: Key[Array, ""] = DOC_KEY0,
    batch_size: int = 8192,
    inside_tol: float = _SDF_INSIDE_TOL,
) -> Array:
    """Pure-JAX rejection sampler for mesh interiors using an SDF/ADF."""
    where_fn = where

    n = int(num_points)
    b = int(batch_size)
    if n < 0:
        raise ValueError(f"num_points must be nonnegative, got {n}.")
    if b <= 0:
        raise ValueError(f"batch_size must be positive, got {b}.")
    if n == 0:
        return jnp.empty((0, 3), dtype=float)

    bounds = jnp.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3):
        raise ValueError(f"bounds must have shape (2, 3), got {bounds.shape}.")
    min_bounds = bounds[0]
    max_bounds = bounds[1]
    span = max_bounds - min_bounds

    inside_tol_arr = jnp.asarray(inside_tol, dtype=float)
    n_arr = jnp.asarray(n, dtype=jnp.int32)

    def _cond(state):
        _, filled, _ = state
        return filled < n_arr

    def _body(state):
        out, filled, k = state
        k, subk = jr.split(k, 2)
        u = _sample_unit_cube(subk, b, dim=3, sampler=sampler)
        pts = min_bounds[None, :] + span[None, :] * u

        sdf = adf(pts)
        mask = sdf < inside_tol_arr
        if where_fn is not None:
            mask = mask & jax.vmap(where_fn)(pts)

        mask_i = mask.astype(jnp.int32)
        csum = jnp.cumsum(mask_i)
        # Positions of accepted points in the output buffer.
        pos = filled + csum - jnp.int32(1)
        # Drop rejected points and anything beyond capacity.
        pos = jnp.where(mask & (pos < n_arr), pos, n_arr)
        out = out.at[pos].set(pts, mode="drop")

        num_accept = csum[-1]
        num_written = jnp.minimum(num_accept, n_arr - filled)
        filled = filled + num_written
        return out, filled, k

    out0 = jnp.zeros((n, 3), dtype=float)
    filled0 = jnp.zeros((), dtype=jnp.int32)
    out, _, _ = jax.lax.while_loop(_cond, _body, (out0, filled0, key))
    return out


def sample_interior_rejection_oneshot_jax(
    adf: Callable[[Array], Array],
    bounds: Array,
    num_points: int,
    *,
    where: Callable[[Array], Bool[Array, ""]] | None = None,
    sampler: str = "latin_hypercube",
    key: Key[Array, ""] = DOC_KEY0,
    batch_size: int = 8192,
    inside_tol: float = _SDF_INSIDE_TOL,
) -> Array:
    """One-shot pure-JAX interior sampler (requires sufficient oversampling)."""
    where_fn = where

    n = int(num_points)
    b = int(batch_size)
    if n < 0:
        raise ValueError(f"num_points must be nonnegative, got {n}.")
    if b <= 0:
        raise ValueError(f"batch_size must be positive, got {b}.")
    if n == 0:
        return jnp.empty((0, 3), dtype=float)

    bounds = jnp.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3):
        raise ValueError(f"bounds must have shape (2, 3), got {bounds.shape}.")
    min_bounds = bounds[0]
    max_bounds = bounds[1]
    span = max_bounds - min_bounds

    u = _sample_unit_cube(key, b, dim=3, sampler=sampler)
    pts = min_bounds[None, :] + span[None, :] * u

    sdf = adf(pts)
    mask = sdf < jnp.asarray(inside_tol, dtype=float)
    if where_fn is not None:
        mask = mask & jax.vmap(where_fn)(pts)

    idx = jnp.nonzero(mask, size=n, fill_value=0)[0]
    return pts[idx]


__all__ = [
    "sample_interior_rejection_jax",
    "sample_interior_rejection_oneshot_jax",
]
