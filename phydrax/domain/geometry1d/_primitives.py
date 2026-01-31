#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Key

from ..._doc import DOC_KEY0
from .._sampling import get_sampler_host, seed_from_key
from ._base import _AbstractGeometry1D


class Interval1d(_AbstractGeometry1D):
    r"""A 1D closed interval geometry.

    Represents the domain

    $$
    \Omega = [a,b]\subset\mathbb{R},
    $$

    with boundary $\partial\Omega = \{a,b\}$ and outward normal $n(a)=-1$, $n(b)=+1$.

    Sampling supports both:
    - `sample_interior`: draws points in $[a,b]$ (optionally filtered by a predicate);
    - `sample_boundary`: draws from the discrete boundary set $\{a,b\}$.
    """

    start: Array
    end: Array
    adf: Callable[[Array], Array]

    def __init__(self, start: ArrayLike, end: ArrayLike):
        start_arr = jnp.asarray(start, dtype=float).reshape(())
        end_arr = jnp.asarray(end, dtype=float).reshape(())
        if bool(start_arr >= end_arr):
            raise ValueError("`start` must be less than `end`.")

        self.start = start_arr
        self.end = end_arr

    @property
    def adf(self) -> Callable[[Array], Array]:
        return self._adf

    @property
    def length(self) -> Array:
        r"""Return the interval length $b-a$."""
        return self.end - self.start

    @property
    def bounds(self) -> Float[Array, "2 1"]:
        return jnp.array([[self.start], [self.end]], dtype=float)

    def equivalent(self, other: object, /) -> bool:
        if not isinstance(other, Interval1d):
            return False
        start_eq = np.isclose(
            np.asarray(self.start),
            np.asarray(other.start),
            rtol=1e-6,
            atol=1e-8,
        )
        end_eq = np.isclose(
            np.asarray(self.end),
            np.asarray(other.end),
            rtol=1e-6,
            atol=1e-8,
        )
        return bool(start_eq) and bool(end_eq)

    def sample_interior(
        self,
        num_points: int,
        *,
        where: Callable | None = None,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        where = where or (lambda _: True)

        def _sample_interior_host(num_points, sampler, where, key):
            rng = np.random.default_rng(seed_from_key(key))
            sampler_fn = get_sampler_host(sampler, dim=1, seed=rng)
            sampled_points = np.empty((0, 1), dtype=float)

            while sampled_points.shape[0] < num_points:
                remaining_points = num_points - sampled_points.shape[0]

                samples = jnp.asarray(sampler_fn(remaining_points), dtype=float)

                if where:
                    # Map samples in [0,1] to [start,end] before applying `where`.
                    pts = samples * (self.end - self.start) + self.start
                    inside = jax.vmap(where)(pts)
                    samples = samples[inside]

                sampled_points = np.vstack((sampled_points, np.asarray(samples)))

            sampled_points = sampled_points[:num_points]
            return sampled_points

        zeros = jnp.zeros((num_points, 1), dtype=float)
        shape_dtype = jax.ShapeDtypeStruct(zeros.shape, zeros.dtype)

        sampled_points = eqx.filter_pure_callback(
            _sample_interior_host,
            num_points,
            sampler,
            where,
            key,
            result_shape_dtypes=shape_dtype,
        )

        return sampled_points * (self.end - self.start) + self.start

    def _sample_interior_separable(
        self,
        num_points: int | Sequence[int],
        *,
        sampler: str = "latin_hypercube",
        where: Callable | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> tuple[tuple[Array, ...], Bool[Array, "..."]]:
        if isinstance(num_points, int):
            num_points_ = num_points
        else:
            num_points_seq = tuple(num_points)
            if len(num_points_seq) != 1:
                raise ValueError(
                    "1D separable sampling expects a single num_points entry."
                )
            num_points_ = int(num_points_seq[0])

        sampled_points = self.sample_interior(
            num_points_,
            sampler=sampler,
            where=where,
            key=key,
        )

        mask = jnp.ones(sampled_points.shape[0], dtype=bool)

        return (sampled_points,), mask

    def sample_boundary(
        self,
        num_points: int,
        *,
        where: Callable | None = None,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        where = where or (lambda _: True)

        def _sample_boundary_host(num_points, sampler, where, key):
            rng = np.random.default_rng(seed_from_key(key))
            sampled_points = np.empty((0, 1), dtype=float)

            while sampled_points.shape[0] < num_points:
                remaining_points = num_points - sampled_points.shape[0]

                choices = np.array([float(self.start), float(self.end)], dtype=float)

                sampled_points_batch = rng.choice(choices, size=(remaining_points, 1))

                if where:
                    pts = jnp.asarray(sampled_points_batch, dtype=float)
                    inside = jax.vmap(where)(pts)
                    pts = pts[inside]
                    sampled_points_batch = np.asarray(pts, dtype=float)

                sampled_points = np.vstack((sampled_points, sampled_points_batch))

            sampled_points = sampled_points[:num_points]
            return sampled_points

        zeros = jnp.zeros((num_points, 1), dtype=float)
        shape_dtype = jax.ShapeDtypeStruct(zeros.shape, zeros.dtype)

        sampled_points = eqx.filter_pure_callback(
            _sample_boundary_host,
            num_points,
            sampler,
            where,
            key,
            result_shape_dtypes=shape_dtype,
        )
        return sampled_points

    def _contains(self, points: Array) -> Bool[Array, " num_points"]:
        pts = jnp.asarray(points, dtype=float)
        a = self.start
        b = self.end
        pts_ = pts[:, 0] if (pts.ndim == 2 and pts.shape[1] == 1) else jnp.squeeze(pts)
        inside = (pts_ >= a) & (pts_ <= b)
        on_boundary = jnp.isclose(pts_, a) | jnp.isclose(pts_, b)
        return inside | on_boundary

    def _on_boundary(self, points: Array) -> Bool[Array, " num_points"]:
        pts = jnp.asarray(points, dtype=float)
        a = self.start
        b = self.end
        pts_ = pts[:, 0] if (pts.ndim == 2 and pts.shape[1] == 1) else jnp.squeeze(pts)
        return jnp.isclose(pts_, a) | jnp.isclose(pts_, b)

    def _boundary_normals(self, points: Array) -> Float[Array, "num_points 1"]:
        pts = jnp.asarray(points, dtype=float)
        a = self.start
        b = self.end
        midpoint = 0.5 * (a + b)
        if pts.ndim == 1:
            pts = jnp.expand_dims(pts, axis=-1)
        return jnp.where(pts < midpoint, -jnp.ones_like(pts), jnp.ones_like(pts))

    def _adf(self, points: Array) -> Array:
        """Signed distance to the interval [start, end].

        Negative inside the interval, 0 on the boundary, positive outside.
        Uses a parabolic surrogate with unit slope at the
        endpoints.
        """
        x = jnp.asarray(points, dtype=float)
        a = self.start
        b = self.end

        # Coerce to 1D for computation
        if x.ndim == 2 and x.shape[1] == 1:
            x1 = x[:, 0]
        else:
            x1 = jnp.squeeze(x)

        # Inside: parabolic surrogate with unit slope at endpoints
        # s(x) = ((x-a)*(x-b)) / (b-a)
        #  - negative for x in (a,b)
        #  - s(a) = s(b) = 0
        #  - s'(a) = -1, s'(b) = 1
        L = b - a

        return ((x1 - a) * (x1 - b)) / L

    def estimate_boundary_subset_measure(
        self,
        where: Callable[[Array], Bool[Array, ""]],
        *,
        num_samples: int = 4096,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        """Estimate subset measure of the 1D boundary endpoints.

        For a 1D interval, the boundary measure is counting measure on {start, end}.
        This method returns 0, 1, or 2 depending on which endpoints satisfy `where`.
        """
        del num_samples, key
        pts = jnp.stack([jnp.atleast_1d(self.start), jnp.atleast_1d(self.end)], axis=0)
        mask = jax.vmap(where)(pts)
        return jnp.sum(mask.astype(float))
