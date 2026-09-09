# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Canonical minimization dominance and bounded exact 2D/3D hypervolume."""

from __future__ import annotations

from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def dominance_matrix(objectives: ArrayLike, valid: ArrayLike | None = None, /) -> Array:
    """Return ``[i, j]`` iff finite valid row i strictly Pareto-dominates row j.

    Equal rows never dominate each other. Archive capacity, duplicate retention,
    stable ordering, front ranks, and crowding remain policies of the caller.
    """
    values = jnp.asarray(objectives)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("objectives must have shape (point, objective).")
    active = _finite_rows(values, valid)
    left, right = values[:, None, :], values[None, :, :]
    return (
        active[:, None]
        & active[None, :]
        & jnp.all(left <= right, axis=-1)
        & jnp.any(left < right, axis=-1)
    )


def nondominated_mask(objectives: ArrayLike, valid: ArrayLike | None = None, /) -> Array:
    """Select every finite valid nondominated minimization row, including ties."""
    values = jnp.asarray(objectives)
    dominance = dominance_matrix(values, valid)
    return _finite_rows(values, valid) & ~jnp.any(dominance, axis=0)


def _finite_rows(values: Array, valid: ArrayLike | None, /) -> Array:
    active = (
        jnp.ones((values.shape[0],), dtype=bool)
        if valid is None
        else jnp.asarray(valid, dtype=bool)
    )
    if active.shape != (values.shape[0],):
        raise ValueError("valid must contain one mask entry per point.")
    return active & jnp.all(jnp.isfinite(values), axis=-1)


def hypervolume(
    objectives: ArrayLike,
    reference: ArrayLike,
    /,
    *,
    valid: ArrayLike | None = None,
    max_points: int = 256,
) -> Array:
    """Exact dominated volume inside a finite minimization reference point.

    The supported profile is exactly two or three objectives and at most
    ``max_points`` input rows (including masked rows). Points outside the
    reference box, nonfinite rows, and masked rows contribute no volume.
    The 3D sweep uses quadratic work and linear temporary storage, not a
    cubic grid or inclusion-exclusion enumeration. This is not a Pareto
    archive truncation: exceeding capacity raises before computing geometry.
    """
    if (
        not isinstance(max_points, Integral)
        or isinstance(max_points, bool)
        or max_points < 1
    ):
        raise ValueError("max_points must be a positive integer.")
    values = jnp.asarray(objectives, dtype=float)
    ref = jnp.asarray(reference, dtype=values.dtype)
    if values.ndim != 2 or values.shape[1] not in (2, 3):
        raise ValueError("Exact hypervolume supports only two or three objectives.")
    if values.shape[0] > max_points:
        raise ValueError("Hypervolume point capacity exceeded.")
    if ref.shape != (values.shape[1],):
        raise ValueError("reference must contain one value per objective.")
    ref = eqx.error_if(ref, jnp.any(~jnp.isfinite(ref)), "reference must be finite.")
    return _hypervolume(values, ref, _finite_rows(values, valid))


def _area_2d(points: Array, reference: Array, valid: Array, /) -> Array:
    clipped = jnp.where(valid[:, None], points, reference)
    order = jnp.argsort(clipped[:, 0], stable=True)
    ordered = clipped[order]
    heights = reference[1] - jax.lax.associative_scan(jnp.minimum, ordered[:, 1])
    right = jnp.concatenate((ordered[1:, 0], reference[:1]))
    return jnp.sum((right - ordered[:, 0]) * heights)


def _hypervolume(points: Array, reference: Array, valid: Array, /) -> Array:
    """Unchecked shape-bounded kernel, suitable for Monte Carlo transformations."""
    count, dimension = points.shape
    if count == 0:
        return jnp.asarray(0.0, dtype=points.dtype)
    active = (
        valid
        & jnp.all(jnp.isfinite(points), axis=1)
        & jnp.all(points < reference, axis=1)
    )
    if dimension == 2:
        return _area_2d(points, reference, active)
    clipped = jnp.where(active[:, None], points, reference)
    ordered = clipped[jnp.argsort(clipped[:, 0], stable=True)]
    # Sort the yz projection once. Each slab only changes its active mask;
    # the cumulative minimum then integrates its exact 2D cross-section.
    yz_order = jnp.argsort(clipped[:, 1], stable=True)
    projected = clipped[yz_order]
    right = jnp.concatenate((ordered[1:, 0], reference[:1]))

    def slab(index, volume):
        selected = projected[:, 0] <= ordered[index, 0]
        z = jnp.where(selected, projected[:, 2], reference[2])
        height = reference[2] - jax.lax.associative_scan(jnp.minimum, z)
        y_right = jnp.concatenate((projected[1:, 1], reference[1:2]))
        area = jnp.sum((y_right - projected[:, 1]) * height)
        return volume + (right[index] - ordered[index, 0]) * area

    return jax.lax.fori_loop(0, count, slab, jnp.asarray(0.0, dtype=points.dtype))


__all__ = ["dominance_matrix", "hypervolume", "nondominated_mask"]
