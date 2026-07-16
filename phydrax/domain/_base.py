#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import functools as ft
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Key

from .._doc import DOC_KEY0
from .._strict import AbstractAttribute, StrictModule
from ._domain import _AbstractUnaryDomain


GeometryTransitionKind: TypeAlias = Literal[
    "unsupported",
    "interval_reflection",
    "box_reflection",
    "implicit_reflection",
    "mesh_walk",
    "chart_retraction",
]


class GeometryTransitionResult(StrictModule):
    """Fixed-shape result of a geometry-constrained coordinate transition."""

    points: Array
    valid: Bool[Array, " num_points"]
    displacement_norm: Float[Array, " num_points"]
    projection_distance: Float[Array, " num_points"]
    reflection_count: Array

    def __init__(
        self,
        points: Array,
        *,
        valid: Array,
        displacement_norm: Array,
        projection_distance: Array,
        reflection_count: Array,
    ):
        pts = jnp.asarray(points, dtype=float)
        if pts.ndim != 2:
            raise ValueError(
                "GeometryTransitionResult.points must have shape (num_points, spatial_dim)."
            )
        n = pts.shape[0]
        valid_arr = jnp.asarray(valid, dtype=bool).reshape((n,))
        displacement_arr = jnp.asarray(displacement_norm, dtype=float).reshape((n,))
        projection_arr = jnp.asarray(projection_distance, dtype=float).reshape((n,))
        reflection_arr = jnp.asarray(reflection_count, dtype=jnp.int32).reshape((n,))
        self.points = pts
        self.valid = valid_arr
        self.displacement_norm = displacement_arr
        self.projection_distance = projection_arr
        self.reflection_count = reflection_arr


class _AbstractGeometry(_AbstractUnaryDomain):
    """Abstract (spatial) geometry."""

    adf: AbstractAttribute[Callable[[Array], Array]]

    @property
    @abstractmethod
    def volume(self) -> Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def spatial_dim(self) -> int:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        return self.spatial_dim

    @property
    def label(self) -> str:
        return "x"

    @property
    def var_dim(self) -> int:
        return int(self.spatial_dim)

    @property
    @abstractmethod
    def bounds(self) -> Float[Array, "2 spatial_dim"]:
        raise NotImplementedError

    @property
    def time(self) -> Literal[False]:
        return False

    @ft.cached_property
    def mesh_bounds(self) -> Float[Array, "2 spatial_dim"]:
        """Axis-aligned bounding box as `[[mins...], [maxs...]]` (raw values)."""
        bounds = jnp.asarray(self.bounds, dtype=float)
        sd = int(self.spatial_dim)
        if bounds.shape != (2, sd):
            raise ValueError(
                f"{type(self).__name__}.bounds must have shape (2, {sd}), got {bounds.shape}."
            )
        return bounds

    @ft.cached_property
    def volume_proportion(self) -> Float[Array, ""]:
        """Fraction of the AABB volume occupied by the geometry (defaults to 1.0)."""
        return jnp.array(1.0, dtype=float)

    @property
    def boundary_measure_value(self) -> Array:
        """Total boundary measure value (boundary length / surface area).

        Concrete geometries should override this where applicable. For 1D geometries,
        this defaults to counting measure on the two endpoints (value = 2).
        """
        if self.spatial_dim == 1:
            return jnp.array(2.0, dtype=float)
        raise NotImplementedError(
            f"{type(self).__name__} must implement `boundary_measure_value`."
        )

    @property
    def interior_transition_kind(self) -> GeometryTransitionKind:
        """Coordinate-transition capability for interior adaptive movement."""
        if self.spatial_dim == 1:
            return "interval_reflection"
        return "implicit_reflection"

    @property
    def boundary_transition_kind(self) -> GeometryTransitionKind:
        """Coordinate-transition capability for boundary adaptive movement."""
        if self.spatial_dim == 1:
            return "unsupported"
        return "chart_retraction"

    def transition_interior(
        self,
        points: Array,
        displacement: Array,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> GeometryTransitionResult:
        """Move interior points while preserving geometry membership."""
        del key
        pts = jnp.asarray(points, dtype=float).reshape((-1, self.spatial_dim))
        delta = jnp.asarray(displacement, dtype=pts.dtype).reshape(pts.shape)
        if self.interior_transition_kind in (
            "interval_reflection",
            "box_reflection",
        ):
            bounds = jnp.asarray(self.bounds, dtype=pts.dtype)
            lower = bounds[0]
            upper = bounds[1]
            span = upper - lower
            proposed = pts + delta
            phase = jnp.mod(proposed - lower, 2.0 * span)
            reflected = lower + jnp.where(
                phase <= span,
                phase,
                2.0 * span - phase,
            )
            below = jnp.ceil(jnp.maximum(lower - proposed, 0.0) / span)
            above = jnp.ceil(jnp.maximum(proposed - upper, 0.0) / span)
            reflection_count = jnp.asarray(
                jnp.sum(below + above, axis=1),
                dtype=jnp.int32,
            )
            return GeometryTransitionResult(
                reflected,
                valid=self._contains(reflected),
                displacement_norm=jnp.linalg.norm(reflected - pts, axis=1),
                projection_distance=jnp.linalg.norm(reflected - proposed, axis=1),
                reflection_count=reflection_count,
            )
        if self.interior_transition_kind != "implicit_reflection":
            raise NotImplementedError(
                f"{type(self).__name__} does not support adaptive interior movement."
            )

        bounds = jnp.asarray(self.bounds, dtype=pts.dtype)
        scale = jnp.max(bounds[1] - bounds[0])
        inset = jnp.maximum(scale * 1e-7, jnp.finfo(pts.dtype).eps)
        current = pts
        remaining = delta
        reflection_count = jnp.zeros((pts.shape[0],), dtype=jnp.int32)

        def adf_batch(x):
            return jax.vmap(self.adf)(x)

        def normal_batch(x):
            gradient = jax.vmap(jax.grad(self.adf))(x)
            norm = jnp.linalg.norm(gradient, axis=1, keepdims=True)
            return gradient / jnp.maximum(norm, jnp.finfo(x.dtype).eps)

        for _ in range(4):
            proposed = current + remaining
            outside = adf_batch(proposed) >= 0.0
            lower = jnp.zeros((pts.shape[0],), dtype=pts.dtype)
            upper = jnp.ones((pts.shape[0],), dtype=pts.dtype)
            for _ in range(12):
                midpoint = 0.5 * (lower + upper)
                trial = current + midpoint[:, None] * remaining
                trial_inside = adf_batch(trial) < 0.0
                lower = jnp.where(trial_inside, midpoint, lower)
                upper = jnp.where(trial_inside, upper, midpoint)
            crossing = current + lower[:, None] * remaining
            normal = normal_batch(crossing)
            tail = (1.0 - lower)[:, None] * remaining
            reflected_tail = tail - 2.0 * jnp.sum(
                tail * normal, axis=1, keepdims=True
            ) * normal
            inside_crossing = crossing - inset * normal
            current = jnp.where(outside[:, None], inside_crossing, proposed)
            remaining = jnp.where(outside[:, None], reflected_tail, 0.0)
            reflection_count = reflection_count + outside.astype(jnp.int32)

        transitioned = current + remaining
        proposed = pts + delta
        for _ in range(4):
            value = adf_batch(transitioned)
            gradient = jax.vmap(jax.grad(self.adf))(transitioned)
            norm_sq = jnp.sum(gradient * gradient, axis=1, keepdims=True)
            correction = jnp.maximum(value + inset, 0.0)[:, None] * gradient
            transitioned = transitioned - correction / jnp.maximum(
                norm_sq,
                jnp.finfo(transitioned.dtype).eps,
            )
        return GeometryTransitionResult(
            transitioned,
            valid=adf_batch(transitioned) < 0.0,
            displacement_norm=jnp.linalg.norm(transitioned - pts, axis=1),
            projection_distance=jnp.linalg.norm(transitioned - proposed, axis=1),
            reflection_count=reflection_count,
        )

    def transition_boundary(
        self,
        points: Array,
        displacement: Array,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> GeometryTransitionResult:
        """Move boundary points tangentially and retract them to the boundary."""
        del key
        if self.boundary_transition_kind != "chart_retraction":
            raise NotImplementedError(
                f"{type(self).__name__} does not support adaptive boundary movement."
            )
        pts = jnp.asarray(points, dtype=float).reshape((-1, self.spatial_dim))
        delta = jnp.asarray(displacement, dtype=pts.dtype).reshape(pts.shape)

        def values(x):
            return jax.vmap(self.adf)(x)

        def gradients(x):
            return jax.vmap(jax.grad(self.adf))(x)

        gradient = gradients(pts)
        norm_sq = jnp.sum(gradient * gradient, axis=1, keepdims=True)
        tangent = delta - (
            jnp.sum(delta * gradient, axis=1, keepdims=True)
            / jnp.maximum(norm_sq, jnp.finfo(pts.dtype).eps)
        ) * gradient
        retracted = pts + tangent
        proposed = retracted
        for _ in range(6):
            value = values(retracted)
            gradient = gradients(retracted)
            norm_sq = jnp.sum(gradient * gradient, axis=1, keepdims=True)
            retracted = retracted - value[:, None] * gradient / jnp.maximum(
                norm_sq,
                jnp.finfo(retracted.dtype).eps,
            )
        gradient = gradients(retracted)
        normal = gradient / jnp.maximum(
            jnp.linalg.norm(gradient, axis=1, keepdims=True),
            jnp.finfo(retracted.dtype).eps,
        )
        bounds = jnp.asarray(self.bounds, dtype=retracted.dtype)
        bracket = 1e-2 * jnp.max(bounds[1] - bounds[0])
        inside = retracted - bracket * normal
        outside = retracted + bracket * normal
        for _ in range(24):
            midpoint = 0.5 * (inside + outside)
            midpoint_inside = self._contains(midpoint)
            inside = jnp.where(midpoint_inside[:, None], midpoint, inside)
            outside = jnp.where(midpoint_inside[:, None], outside, midpoint)
        retracted = inside
        return GeometryTransitionResult(
            retracted,
            valid=self._on_boundary(retracted),
            displacement_norm=jnp.linalg.norm(retracted - pts, axis=1),
            projection_distance=jnp.linalg.norm(retracted - proposed, axis=1),
            reflection_count=jnp.zeros((pts.shape[0],), dtype=jnp.int32),
        )

    @abstractmethod
    def estimate_boundary_subset_measure(
        self,
        where: Callable[[Array], Bool[Array, ""]],
        *,
        num_samples: int = 4096,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        """Estimate boundary subset measure of {x: where(x)=True}."""
        raise NotImplementedError

    def _check_points_on_boundary(self, points: Array) -> Array:
        return eqx.error_if(
            points,
            pred=~self._on_boundary(points),
            msg="All points must be on the boundary of the domain.",
        )

    @abstractmethod
    def sample_interior(
        self,
        num_points: int,
        *,
        where: Callable | None = None,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def sample_boundary(
        self,
        num_points: int,
        *,
        where: Callable | None = None,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _sample_interior_separable(
        self,
        num_points: int | Sequence[int],
        *,
        sampler: str = "latin_hypercube",
        where: Callable | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> tuple[tuple[Array, ...], Bool[Array, "..."]]:
        """Internal helper for separable interior sampling."""
        raise NotImplementedError

    @abstractmethod
    def _contains(self, points: Array) -> Bool[Array, " num_points"]:
        raise NotImplementedError

    @abstractmethod
    def _on_boundary(self, points: Array) -> Bool[Array, " num_points"]:
        raise NotImplementedError

    @abstractmethod
    def _boundary_normals(self, points: Array) -> Float[Array, "num_points spatial_dim"]:
        raise NotImplementedError
