#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.domain import ProductStructure
from phydrax.domain._base import _AbstractGeometry


class Box3D(_AbstractGeometry):
    min_corner: jax.Array
    max_corner: jax.Array
    adf: Callable

    def __init__(self, min_corner=(-1.0, -1.0, -1.0), max_corner=(1.0, 1.0, 1.0)):
        self.min_corner = jnp.asarray(min_corner, dtype=float)
        self.max_corner = jnp.asarray(max_corner, dtype=float)

    @property
    def adf(self):
        return self._adf

    @property
    def spatial_dim(self) -> int:
        return 3

    @property
    def bounds(self) -> jax.Array:
        return jnp.stack([self.min_corner, self.max_corner], axis=0)

    @property
    def volume(self) -> jax.Array:
        ext = self.max_corner - self.min_corner
        return jnp.prod(ext)

    @property
    def boundary_measure_value(self) -> jax.Array:
        dx, dy, dz = (self.max_corner - self.min_corner).tolist()
        return jnp.array(2.0 * (dx * dy + dx * dz + dy * dz), dtype=float)

    def estimate_boundary_subset_measure(self, where, *, num_samples=4096, key=jr.key(0)):
        del where, num_samples, key
        return self.boundary_measure_value

    def sample_interior(
        self,
        num_points,
        *,
        where=None,
        sampler="latin_hypercube",
        key=jr.key(0),
    ):
        del where, sampler
        key = jr.key(0) if key is None else key
        u = jr.uniform(key, (int(num_points), 3), minval=0.0, maxval=1.0)
        return self.min_corner[None, :] + u * (self.max_corner - self.min_corner)[None, :]

    def sample_boundary(
        self,
        num_points,
        *,
        where=None,
        sampler="latin_hypercube",
        key=jr.key(0),
    ):
        del where, sampler
        key = jr.key(0) if key is None else key
        key_face, key_u = jr.split(key, 2)
        faces = jr.randint(key_face, (int(num_points),), 0, 6)
        u = jr.uniform(key_u, (int(num_points), 2), minval=0.0, maxval=1.0)
        lo = self.min_corner
        hi = self.max_corner
        pts = jnp.zeros((int(num_points), 3), dtype=float)
        x = lo[0] + u[:, 0] * (hi[0] - lo[0])
        y = lo[1] + u[:, 1] * (hi[1] - lo[1])
        z = lo[2] + u[:, 0] * (hi[2] - lo[2])

        pts = pts.at[:, 0].set(x)
        pts = pts.at[:, 1].set(y)
        pts = pts.at[:, 2].set(z)
        pts = jnp.where(faces[:, None] == 0, pts.at[:, 0].set(lo[0]), pts)
        pts = jnp.where(faces[:, None] == 1, pts.at[:, 0].set(hi[0]), pts)
        pts = jnp.where(faces[:, None] == 2, pts.at[:, 1].set(lo[1]), pts)
        pts = jnp.where(faces[:, None] == 3, pts.at[:, 1].set(hi[1]), pts)
        pts = jnp.where(faces[:, None] == 4, pts.at[:, 2].set(lo[2]), pts)
        pts = jnp.where(faces[:, None] == 5, pts.at[:, 2].set(hi[2]), pts)
        return pts

    def _sample_interior_separable(
        self,
        num_points,
        *,
        sampler="latin_hypercube",
        where=None,
        key=jr.key(0),
    ):
        del sampler, where
        if isinstance(num_points, int):
            counts = (int(num_points),) * 3
        else:
            counts = tuple(int(n) for n in num_points)
            if len(counts) != 3:
                raise ValueError("Box3D._sample_interior_separable expects 3 counts.")

        k0, k1, k2 = jr.split(key, 3)
        xs = jr.uniform(
            k0,
            (counts[0],),
            minval=float(self.min_corner[0]),
            maxval=float(self.max_corner[0]),
        )
        ys = jr.uniform(
            k1,
            (counts[1],),
            minval=float(self.min_corner[1]),
            maxval=float(self.max_corner[1]),
        )
        zs = jr.uniform(
            k2,
            (counts[2],),
            minval=float(self.min_corner[2]),
            maxval=float(self.max_corner[2]),
        )
        mask = jnp.ones((counts[0], counts[1], counts[2]), dtype=bool)
        return (xs, ys, zs), mask

    def _contains(self, points):
        pts = jnp.asarray(points, dtype=float)
        lo = self.min_corner
        hi = self.max_corner
        inside = (pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0])
        inside = inside & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1])
        inside = inside & (pts[:, 2] >= lo[2]) & (pts[:, 2] <= hi[2])
        return inside

    def _on_boundary(self, points):
        pts = jnp.asarray(points, dtype=float)
        lo = self.min_corner
        hi = self.max_corner
        eps = 1e-6
        on = jnp.isclose(pts[:, 0], lo[0], atol=eps) | jnp.isclose(
            pts[:, 0], hi[0], atol=eps
        )
        on = (
            on
            | jnp.isclose(pts[:, 1], lo[1], atol=eps)
            | jnp.isclose(pts[:, 1], hi[1], atol=eps)
        )
        on = (
            on
            | jnp.isclose(pts[:, 2], lo[2], atol=eps)
            | jnp.isclose(pts[:, 2], hi[2], atol=eps)
        )
        return on

    def _boundary_normals(self, points):
        pts = jnp.asarray(points, dtype=float)
        lo = self.min_corner
        hi = self.max_corner
        eps = 1e-6
        n = jnp.zeros_like(pts)
        n = jnp.where(
            jnp.isclose(pts[:, 0:1], lo[0], atol=eps), n.at[:, 0:1].set(-1.0), n
        )
        n = jnp.where(jnp.isclose(pts[:, 0:1], hi[0], atol=eps), n.at[:, 0:1].set(1.0), n)
        n = jnp.where(
            jnp.isclose(pts[:, 1:2], lo[1], atol=eps), n.at[:, 1:2].set(-1.0), n
        )
        n = jnp.where(jnp.isclose(pts[:, 1:2], hi[1], atol=eps), n.at[:, 1:2].set(1.0), n)
        n = jnp.where(
            jnp.isclose(pts[:, 2:3], lo[2], atol=eps), n.at[:, 2:3].set(-1.0), n
        )
        n = jnp.where(jnp.isclose(pts[:, 2:3], hi[2], atol=eps), n.at[:, 2:3].set(1.0), n)
        return n

    def _adf(self, points):
        pts = jnp.asarray(points, dtype=float)
        lo = self.min_corner
        hi = self.max_corner
        c = jnp.clip(pts, lo, hi)
        d_out = jnp.linalg.norm(pts - c, axis=-1)
        inside = self._contains(pts)
        d_in = -jnp.min(
            jnp.stack(
                [
                    pts[:, 0] - lo[0],
                    hi[0] - pts[:, 0],
                    pts[:, 1] - lo[1],
                    hi[1] - pts[:, 1],
                    pts[:, 2] - lo[2],
                    hi[2] - pts[:, 2],
                ],
                axis=0,
            ),
            axis=0,
        )
        return jnp.where(inside, d_in, d_out)

    def equivalent(self, other: object, /) -> bool:
        if not isinstance(other, Box3D):
            return False
        lo_eq = np.allclose(
            np.asarray(self.min_corner),
            np.asarray(other.min_corner),
            rtol=1e-6,
            atol=1e-8,
        )
        hi_eq = np.allclose(
            np.asarray(self.max_corner),
            np.asarray(other.max_corner),
            rtol=1e-6,
            atol=1e-8,
        )
        return bool(lo_eq) and bool(hi_eq)


@pytest.fixture
def box3d():
    return Box3D()


@pytest.fixture
def sample_batch():
    def _sample(component, /, *, blocks, num_points, key=0, sampler="latin_hypercube"):
        structure = ProductStructure(blocks=blocks)
        return component.sample(
            num_points,
            structure=structure,
            sampler=sampler,
            key=jr.key(int(key)),
        )

    return _sample


@pytest.fixture
def sample_coord_separable():
    def _sample(
        component,
        coord_separable,
        /,
        *,
        num_points=(),
        dense_blocks=(),
        key=0,
        sampler="latin_hypercube",
    ):
        dense_structure = (
            ProductStructure(blocks=dense_blocks) if dense_blocks is not None else None
        )
        return component.sample_coord_separable(
            coord_separable,
            num_points=num_points,
            dense_structure=dense_structure,
            sampler=sampler,
            key=jr.key(int(key)),
        )

    return _sample
