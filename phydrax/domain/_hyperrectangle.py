#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Key

from .._doc import DOC_KEY0
from .._sampling import get_sampler_host, seed_from_key
from ._base import AbstractGeometry, EnforcementGateMethod, GeometryTransitionKind
from ._grid import broadcasted_grid
from ._structure import _validate_label


class HyperRectangle(AbstractGeometry):
    r"""Axis-aligned hyperrectangle in R^d.

    Represents the set

    ```text
    Omega = {x in R^d : lower_i <= x_i <= upper_i for every i}
    ```

    This is the natural geometry for vector-valued feature spaces, parameter boxes,
    and tabular supervised learning problems where each row is one point in R^d.
    """

    lower: Array
    upper: Array
    _label: str
    adf: Callable[[Array], Array]

    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        *,
        label: str = "x",
    ):
        lower_arr = jnp.asarray(lower, dtype=float)
        upper_arr = jnp.asarray(upper, dtype=float)
        if lower_arr.ndim != 1 or upper_arr.ndim != 1:
            raise ValueError("HyperRectangle lower/upper must be one-dimensional.")
        if lower_arr.shape != upper_arr.shape:
            raise ValueError(
                "HyperRectangle lower and upper must have matching shapes; "
                f"got {lower_arr.shape} and {upper_arr.shape}."
            )
        if int(lower_arr.shape[0]) <= 0:
            raise ValueError("HyperRectangle dimension must be positive.")
        if bool(jnp.any(upper_arr <= lower_arr)):
            raise ValueError("HyperRectangle requires upper > lower in every dimension.")

        _validate_label(label)

        self.lower = lower_arr
        self.upper = upper_arr
        self._label = str(label)

    @property
    def label(self) -> str:
        return self._label

    @property
    def adf(self) -> Callable[[Array], Array]:
        return self._adf

    @property
    def _enforcement_gate_builder(
        self,
    ) -> Callable[..., Callable[[Array], Array]]:
        return self._make_enforcement_gate

    def _make_enforcement_gate(
        self,
        *,
        method: EnforcementGateMethod = "auto",
        saturation_fraction: float = 0.5,
        linear_fraction: float = 0.5,
    ) -> Callable[[Array], Array]:
        """Build the analytic dimensionless product gate for a box."""
        widths = self.upper - self.lower

        def gate(points: Array) -> Array:
            pts, single = self._points_2d(points)
            axis_gates = 4.0 * (pts - self.lower) * (self.upper - pts) / (widths * widths)
            values = jnp.prod(axis_gates, axis=-1)
            return values[0] if single else values

        return jax.jit(gate)

    @property
    def spatial_dim(self) -> int:
        return int(self.lower.shape[0])

    @property
    def interior_transition_kind(self) -> GeometryTransitionKind:
        return "box_reflection"

    @property
    def bounds(self) -> Float[Array, "2 spatial_dim"]:
        return jnp.stack((self.lower, self.upper), axis=0)

    @property
    def volume(self) -> Array:
        return jnp.prod(self.upper - self.lower)

    @property
    def boundary_measure_value(self) -> Array:
        widths = self.upper - self.lower
        if int(widths.shape[0]) == 1:
            return jnp.asarray(2.0, dtype=float)
        face_measures = self.volume / widths
        return 2.0 * jnp.sum(face_measures)

    def _same_factor_support(self, other: object, /) -> bool:
        if not isinstance(other, HyperRectangle):
            return False
        if self.spatial_dim != other.spatial_dim:
            return False
        return bool(
            np.allclose(np.asarray(self.lower), np.asarray(other.lower))
            and np.allclose(np.asarray(self.upper), np.asarray(other.upper))
        )

    def _points_2d(self, points: ArrayLike, /) -> tuple[Array, bool]:
        pts = jnp.asarray(points, dtype=float)
        dim = int(self.spatial_dim)

        if pts.ndim == 0:
            if dim != 1:
                raise ValueError(f"Expected points with trailing dimension {dim}.")
            return pts.reshape((1, 1)), True

        if pts.ndim == 1:
            if dim == 1:
                return pts.reshape((-1, 1)), int(pts.shape[0]) == 1
            if int(pts.shape[0]) != dim:
                raise ValueError(f"Expected point with shape ({dim},), got {pts.shape}.")
            return pts.reshape((1, dim)), True

        if pts.ndim == 2 and int(pts.shape[1]) == dim:
            return pts, False

        raise ValueError(
            f"Expected points with shape ({dim},) or (N, {dim}), got {pts.shape}."
        )

    def sample_interior(
        self,
        num_points: int,
        *,
        where: Callable | None = None,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        dim = int(self.spatial_dim)
        where_fn = where or (lambda _: True)

        def _sample_interior_host(num_points, sampler, where_fn, key):
            rng = np.random.default_rng(seed_from_key(key))
            sampler_fn = get_sampler_host(sampler, dim=dim, seed=rng)
            sampled = np.empty((0, dim), dtype=float)

            while sampled.shape[0] < int(num_points):
                remaining = int(num_points) - sampled.shape[0]
                unit = sampler_fn(remaining)
                points = lower + unit * (upper - lower)
                if where_fn is not None:
                    mask = np.asarray(
                        jax.vmap(where_fn)(jnp.asarray(points, dtype=float)),
                        dtype=bool,
                    ).reshape((-1,))
                    points = points[mask]
                sampled = np.vstack((sampled, np.asarray(points, dtype=float)))

            return sampled[: int(num_points)]

        zeros = jnp.zeros((int(num_points), dim), dtype=float)
        shape_dtype = jax.ShapeDtypeStruct(zeros.shape, zeros.dtype)
        return eqx.filter_pure_callback(
            _sample_interior_host,
            int(num_points),
            sampler,
            where_fn,
            key,
            result_shape_dtypes=shape_dtype,
        )

    def sample_boundary(
        self,
        num_points: int,
        *,
        where: Callable | None = None,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        widths = upper - lower
        dim = int(self.spatial_dim)
        where_fn = where or (lambda _: True)

        if dim == 1:
            face_probs = np.asarray([0.5, 0.5], dtype=float)
        else:
            face_measures = np.prod(widths) / widths
            face_probs = np.repeat(face_measures, 2)
            face_probs = face_probs / np.sum(face_probs)

        def _sample_boundary_host(num_points, sampler, where_fn, key):
            rng = np.random.default_rng(seed_from_key(key))
            sampler_dim = max(dim - 1, 1)
            sampler_fn = get_sampler_host(sampler, dim=sampler_dim, seed=rng)
            sampled = np.empty((0, dim), dtype=float)

            while sampled.shape[0] < int(num_points):
                remaining = int(num_points) - sampled.shape[0]
                face_ids = rng.choice(2 * dim, size=(remaining,), p=face_probs)
                axes = face_ids // 2
                sides = face_ids % 2
                unit = sampler_fn(remaining)
                points = np.empty((remaining, dim), dtype=float)

                for row in range(remaining):
                    axis = int(axes[row])
                    local_col = 0
                    for col in range(dim):
                        if col == axis:
                            points[row, col] = (
                                lower[col] if sides[row] == 0 else upper[col]
                            )
                        else:
                            points[row, col] = (
                                lower[col] + unit[row, local_col] * widths[col]
                            )
                            local_col += 1

                if where_fn is not None:
                    mask = np.asarray(
                        jax.vmap(where_fn)(jnp.asarray(points, dtype=float)),
                        dtype=bool,
                    ).reshape((-1,))
                    points = points[mask]
                sampled = np.vstack((sampled, np.asarray(points, dtype=float)))

            return sampled[: int(num_points)]

        zeros = jnp.zeros((int(num_points), dim), dtype=float)
        shape_dtype = jax.ShapeDtypeStruct(zeros.shape, zeros.dtype)
        return eqx.filter_pure_callback(
            _sample_boundary_host,
            int(num_points),
            sampler,
            where_fn,
            key,
            result_shape_dtypes=shape_dtype,
        )

    def _sample_interior_separable(
        self,
        num_points: int | Sequence[int],
        *,
        sampler: str = "latin_hypercube",
        where: Callable | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> tuple[tuple[Array, ...], Bool[Array, "..."]]:
        dim = int(self.spatial_dim)
        if isinstance(num_points, int):
            counts = (int(num_points),) * dim
        else:
            counts = tuple(int(n) for n in num_points)
            if len(counts) != dim:
                raise ValueError(
                    f"HyperRectangle separable sampling expects {dim} counts, got {len(counts)}."
                )

        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)

        def _sample_axes_host(counts, sampler, key):
            rng = np.random.default_rng(seed_from_key(key))
            axes = []
            for i, n in enumerate(counts):
                sampler_fn = get_sampler_host(sampler, dim=1, seed=rng)
                unit = sampler_fn(int(n)).reshape((int(n),))
                axes.append(lower[i] + unit * (upper[i] - lower[i]))
            return tuple(np.asarray(axis, dtype=float) for axis in axes)

        result_shape = tuple(
            jax.ShapeDtypeStruct((int(n),), np.dtype(float)) for n in counts
        )
        coords = eqx.filter_pure_callback(
            _sample_axes_host,
            counts,
            sampler,
            key,
            result_shape_dtypes=result_shape,
        )
        coords = tuple(jnp.asarray(c, dtype=float) for c in coords)

        if where is None:
            mask = jnp.ones(tuple(counts), dtype=bool)
        else:
            grid = broadcasted_grid(coords)
            pts = grid.reshape((-1, dim))
            mask = jax.vmap(where)(pts).reshape(tuple(counts))
            mask = jnp.asarray(mask, dtype=bool)

        return coords, mask

    def _contains(self, points: Array) -> Bool[Array, " num_points"]:
        pts, _ = self._points_2d(points)
        return jnp.all((pts >= self.lower) & (pts <= self.upper), axis=-1)

    def _on_boundary(self, points: Array) -> Bool[Array, " num_points"]:
        pts, _ = self._points_2d(points)
        inside = self._contains(pts)
        lower_face = jnp.isclose(pts, self.lower)
        upper_face = jnp.isclose(pts, self.upper)
        return inside & jnp.any(lower_face | upper_face, axis=-1)

    def _boundary_normals(self, points: Array) -> Float[Array, "num_points spatial_dim"]:
        pts, _ = self._points_2d(points)
        lower_face = jnp.isclose(pts, self.lower)
        upper_face = jnp.isclose(pts, self.upper)
        raw = upper_face.astype(float) - lower_face.astype(float)
        raw_norm = jnp.linalg.norm(raw, axis=-1, keepdims=True)

        dist_lower = jnp.abs(pts - self.lower)
        dist_upper = jnp.abs(pts - self.upper)
        face_dist = jnp.concatenate((dist_lower, dist_upper), axis=-1)
        nearest = jnp.argmin(face_dist, axis=-1)
        dim = int(self.spatial_dim)
        nearest_axis = nearest % dim
        nearest_sign = jnp.where(nearest < dim, -1.0, 1.0)
        fallback = jax.nn.one_hot(nearest_axis, dim, dtype=float) * nearest_sign[:, None]

        safe_raw = raw / jnp.maximum(raw_norm, jnp.finfo(float).eps)
        normals = jnp.where(raw_norm > 0.0, safe_raw, fallback)
        norm = jnp.linalg.norm(normals, axis=-1, keepdims=True)
        return normals / jnp.maximum(norm, jnp.finfo(float).eps)

    def _adf(self, points: Array) -> Array:
        pts, single = self._points_2d(points)
        center = 0.5 * (self.lower + self.upper)
        half_width = 0.5 * (self.upper - self.lower)
        q = jnp.abs(pts - center) - half_width
        outside = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
        inside = jnp.minimum(jnp.max(q, axis=-1), 0.0)
        sdf = outside + inside
        return sdf[0] if single else sdf

    def estimate_boundary_subset_measure(
        self,
        where: Callable[[Array], Bool[Array, ""]],
        *,
        num_samples: int = 4096,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        pts = self.sample_boundary(int(num_samples), key=key)
        mask = jax.vmap(where)(pts)
        return jnp.mean(jnp.asarray(mask, dtype=float)) * self.boundary_measure_value


__all__ = ["HyperRectangle"]
