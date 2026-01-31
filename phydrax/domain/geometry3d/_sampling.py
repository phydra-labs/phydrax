#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Bool, Key

from ..._doc import DOC_KEY0
from .._sampling import get_sampler_host, seed_from_key


_SDF_INSIDE_TOL = -1e-8


def _sample_barycentric_coords(
    self, num_points: int, sampler: str, key: Key[Array, ""]
) -> Array:
    """Sample random barycentric coordinates."""
    rng = np.random.default_rng(seed_from_key(key))
    sampler_fn = get_sampler_host(sampler, dim=2, seed=rng)
    uv = jnp.asarray(sampler_fn(int(num_points)), dtype=float)
    u = uv[:, :1]
    v = uv[:, 1:2]

    mask = (u + v) > 1.0
    u = jnp.where(mask, 1.0 - u, u)
    v = jnp.where(mask, 1.0 - v, v)
    w = 1.0 - (u + v)
    return jnp.hstack([u, v, w])


def sample_boundary(
    self,
    num_points: int,
    *,
    where: Callable | None = None,
    sampler: str = "latin_hypercube",
    key: Key[Array, ""] = DOC_KEY0,
) -> Array:
    where_fn = where

    def _sample_boundary_host(num_points, sampler, where_fn, key):
        batches: list[np.ndarray] = []
        total = 0

        while total < num_points:
            remaining_points = num_points - total
            tkey, bkey, key = jr.split(key, 3)

            sampled_triangle_indices = jr.choice(
                tkey,
                self.mesh_faces.shape[0],
                shape=(remaining_points,),
                p=self.triangle_probs,
            )
            sampled_triangles = self.mesh_faces[sampled_triangle_indices]

            barycentric_coords = _sample_barycentric_coords(
                self, remaining_points, sampler, bkey
            )

            vertices = self.mesh_vertices[sampled_triangles]
            sampled_points_batch = jnp.sum(
                vertices * barycentric_coords[..., jnp.newaxis],
                axis=1,
            )

            if where_fn is not None:
                inside = jax.vmap(where_fn)(sampled_points_batch)
                sampled_points_batch = sampled_points_batch[inside]

            batch_np = np.asarray(sampled_points_batch, dtype=float)
            if batch_np.size:
                batches.append(batch_np)
                total += batch_np.shape[0]

        if num_points == 0:
            return np.empty((0, 3), dtype=float)
        return np.concatenate(batches, axis=0)[:num_points]

    zeros = jnp.zeros((num_points, 3), dtype=float)
    shape_dtype = jax.ShapeDtypeStruct(zeros.shape, zeros.dtype)

    sampled_points = eqx.filter_pure_callback(
        _sample_boundary_host,
        num_points,
        sampler,
        where_fn,
        key,
        result_shape_dtypes=shape_dtype,
    )
    return sampled_points


def sample_interior(
    self,
    num_points: int,
    *,
    where: Callable | None = None,
    sampler: str = "latin_hypercube",
    key: Key[Array, ""] = DOC_KEY0,
) -> Array:
    where_fn = where

    from ._sdf import make_mesh_sdf_fast

    sdf_fast = make_mesh_sdf_fast(self)

    def _sample_interior_host(num_points, sampler, where_fn, key):
        rng = np.random.default_rng(seed_from_key(key))
        sampler_fn = get_sampler_host(sampler, dim=3, seed=rng)
        min_bounds, max_bounds = np.asarray(self.mesh_bounds, dtype=float)
        oversampling_factor = max(1, int(1.0 / self.volume_proportion))
        batches: list[np.ndarray] = []
        total = 0

        while total < num_points:
            remaining_points = num_points - total

            num_samples = oversampling_factor * remaining_points
            samples = sampler_fn(num_samples)
            samples = min_bounds + (max_bounds - min_bounds) * samples

            sdf = np.asarray(sdf_fast(jnp.asarray(samples, dtype=float)), dtype=float)
            inside = sdf < float(_SDF_INSIDE_TOL)
            if where_fn is not None:
                inside = inside & np.asarray(
                    jax.vmap(where_fn)(jnp.asarray(samples, dtype=float)),
                    dtype=bool,
                )
            samples = np.asarray(samples[inside], dtype=float)
            if samples.size:
                batches.append(samples)
                total += samples.shape[0]

        if num_points == 0:
            return np.empty((0, 3), dtype=float)
        return np.concatenate(batches, axis=0)[:num_points]

    zeros = jnp.zeros((num_points, 3), dtype=float)
    shape_dtype = jax.ShapeDtypeStruct(zeros.shape, zeros.dtype)

    sampled_points = eqx.filter_pure_callback(
        _sample_interior_host,
        num_points,
        sampler,
        where_fn,
        key,
        result_shape_dtypes=shape_dtype,
    )
    return sampled_points


def _sample_interior_separable(
    self,
    num_points: Sequence[int],
    *,
    sampler: str = "latin_hypercube",
    where: Callable | None = None,
    add_delta: bool = False,
    key: Key[Array, ""] = DOC_KEY0,
) -> tuple[
    tuple[Array, Array, Array],
    Bool[Array, "num_points_x num_points_y num_points_z"],
]:
    """Sample a separable Cartesian grid with a mask for interior points."""
    num_points_x, num_points_y, num_points_z = num_points

    def _sample_interior_separable_host(
        num_points_x, num_points_y, num_points_z, sampler, where, key
    ):
        bounds = np.asarray(self.bounds, dtype=float)
        min_bounds = bounds[0]
        max_bounds = bounds[1]
        xmin, ymin, zmin = min_bounds
        xmax, ymax, zmax = max_bounds

        if add_delta:
            delta = 1e-2
            xmin = xmin - delta
            xmax = xmax + delta
            ymin = ymin - delta
            ymax = ymax + delta
            zmin = zmin - delta
            zmax = zmax + delta

        ss = np.random.SeedSequence(seed_from_key(key))
        xrng, yrng, zrng = (np.random.default_rng(s) for s in ss.spawn(3))
        sampler_x = get_sampler_host(sampler, dim=1, seed=xrng)
        sampler_y = get_sampler_host(sampler, dim=1, seed=yrng)
        sampler_z = get_sampler_host(sampler, dim=1, seed=zrng)

        sampled_x = (
            jnp.asarray(sampler_x(num_points_x), dtype=float).squeeze() * (xmax - xmin)
            + xmin
        )
        sampled_y = (
            jnp.asarray(sampler_y(num_points_y), dtype=float).squeeze() * (ymax - ymin)
            + ymin
        )
        sampled_z = (
            jnp.asarray(sampler_z(num_points_z), dtype=float).squeeze() * (zmax - zmin)
            + zmin
        )

        grid_x, grid_y, grid_z = jnp.meshgrid(
            sampled_x, sampled_y, sampled_z, indexing="ij"
        )
        grid = jnp.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))

        mask = self.adf_orig(grid) < _SDF_INSIDE_TOL
        if where:
            mask = mask & jax.vmap(where)(grid)

        mask = mask.reshape((sampled_x.size, sampled_y.size, sampled_z.size))
        return (sampled_x, sampled_y, sampled_z), mask

    zeros_x = jnp.zeros(num_points_x, dtype=float)
    zeros_y = jnp.zeros(num_points_y, dtype=float)
    zeros_z = jnp.zeros(num_points_z, dtype=float)
    zeros_mask = jnp.zeros((num_points_x, num_points_y, num_points_z), dtype=bool)

    shape_dtype_x = jax.ShapeDtypeStruct(zeros_x.shape, zeros_x.dtype)
    shape_dtype_y = jax.ShapeDtypeStruct(zeros_y.shape, zeros_y.dtype)
    shape_dtype_z = jax.ShapeDtypeStruct(zeros_z.shape, zeros_z.dtype)
    shape_dtype_mask = jax.ShapeDtypeStruct(zeros_mask.shape, zeros_mask.dtype)

    (sampled_x, sampled_y, sampled_z), mask = eqx.filter_pure_callback(
        _sample_interior_separable_host,
        num_points_x,
        num_points_y,
        num_points_z,
        sampler,
        where,
        key,
        result_shape_dtypes=(
            (shape_dtype_x, shape_dtype_y, shape_dtype_z),
            shape_dtype_mask,
        ),
    )
    return (sampled_x, sampled_y, sampled_z), mask


__all__ = [
    "_sample_barycentric_coords",
    "_sample_interior_separable",
    "sample_boundary",
    "sample_interior",
]
