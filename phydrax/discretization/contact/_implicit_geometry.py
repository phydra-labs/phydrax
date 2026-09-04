#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._guarantee import ContactGuaranteeLevel
from ._surface import CollisionFeaturePolicy


def _analytic_feature_policy(
    feature_policy: CollisionFeaturePolicy, /
) -> CollisionFeaturePolicy:
    if not isinstance(feature_policy, CollisionFeaturePolicy):
        raise TypeError("feature_policy must be CollisionFeaturePolicy.")
    if (
        feature_policy.vertex_count != 0
        or feature_policy.edge_count != 0
        or feature_policy.face_count != 0
        or feature_policy.analytic_count != 1
    ):
        raise ValueError(
            "Implicit contact geometry requires exactly one analytic feature."
        )
    return feature_policy


class ImplicitContactEvaluation(StrictModule):
    signed_distance: Array
    normal: Array
    closest_point: Array
    feature_margin: Array
    guarantee_level: Array
    finite: Array
    successful: Array
    geometry_id: str = eqx.field(static=True)


class AbstractImplicitContactGeometry(StrictModule, NonTrainableState):
    @property
    @abc.abstractmethod
    def ambient_dimension(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def geometry_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def feature_policy(self) -> CollisionFeaturePolicy:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def guarantee_level(self) -> ContactGuaranteeLevel:
        raise NotImplementedError

    @abc.abstractmethod
    def signed_distance(self, points: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def normal(self, points: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def bounds(self, /) -> tuple[Array, Array]:
        raise NotImplementedError

    def evaluate(self, points: ArrayLike, /) -> ImplicitContactEvaluation:
        points_ = jnp.asarray(points)
        if points_.ndim < 1 or points_.shape[-1] != self.ambient_dimension:
            raise ValueError("Implicit contact query has invalid trailing dimension.")
        distance = self.signed_distance(points_)
        normal = self.normal(points_)
        closest = points_ - distance[..., None] * normal
        normal_norm = jnp.sqrt(jnp.sum(normal * normal, axis=-1))
        finite = (
            jnp.all(jnp.isfinite(distance))
            & jnp.all(jnp.isfinite(normal))
            & jnp.all(jnp.isfinite(closest))
        )
        unit = jnp.all(jnp.abs(normal_norm - 1.0) <= 64.0 * jnp.finfo(points_.dtype).eps)
        return ImplicitContactEvaluation(
            distance,
            normal,
            closest,
            jnp.abs(distance),
            jnp.asarray(int(self.guarantee_level), dtype=jnp.int32),
            finite,
            finite & unit,
            self.geometry_id,
        )


class SphereContactGeometry(AbstractImplicitContactGeometry):
    center: Array
    radius: float = eqx.field(static=True)
    _feature_policy: CollisionFeaturePolicy
    _geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        center: ArrayLike,
        radius: float,
        /,
        *,
        feature_policy: CollisionFeaturePolicy,
    ):
        center_ = jnp.asarray(center)
        radius_ = float(radius)
        if center_.shape not in ((2,), (3,)):
            raise ValueError("Sphere center requires dimension two or three.")
        if not np.isfinite(radius_) or radius_ <= 0.0:
            raise ValueError("Sphere radius must be finite and positive.")
        features = _analytic_feature_policy(feature_policy)
        self.center = center_
        self.radius = radius_
        self._feature_policy = features
        self._geometry_id = canonical_fingerprint(
            {
                "kind": "sphere-contact-geometry",
                "center": array_tree_fingerprint(np.asarray(center_)),
                "dimension": int(center_.size),
                "radius": radius_.hex(),
                "feature_policy": features.policy_id,
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return int(self.center.size)

    @property
    def geometry_id(self) -> str:
        return self._geometry_id

    @property
    def feature_policy(self) -> CollisionFeaturePolicy:
        return self._feature_policy

    @property
    def guarantee_level(self) -> ContactGuaranteeLevel:
        return ContactGuaranteeLevel.ANALYTIC_CONSERVATIVE

    def signed_distance(self, points: ArrayLike, /) -> Array:
        relative = jnp.asarray(points, dtype=self.center.dtype) - self.center
        return jnp.sqrt(jnp.sum(relative * relative, axis=-1)) - self.radius

    def normal(self, points: ArrayLike, /) -> Array:
        relative = jnp.asarray(points, dtype=self.center.dtype) - self.center
        norm = jnp.sqrt(jnp.sum(relative * relative, axis=-1, keepdims=True))
        return relative / jnp.maximum(norm, jnp.finfo(relative.dtype).eps)

    def bounds(self, /) -> tuple[Array, Array]:
        return self.center - self.radius, self.center + self.radius


class PlaneContactGeometry(AbstractImplicitContactGeometry):
    unit_normal: Array
    offset: float = eqx.field(static=True)
    _feature_policy: CollisionFeaturePolicy
    _geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        normal: ArrayLike,
        offset: float,
        /,
        *,
        feature_policy: CollisionFeaturePolicy,
    ):
        normal_ = jnp.asarray(normal)
        if normal_.shape not in ((2,), (3,)):
            raise ValueError("Plane normal requires dimension two or three.")
        norm = jnp.sqrt(jnp.sum(normal_ * normal_))
        if not bool(jnp.isfinite(norm)) or norm <= 0.0:
            raise ValueError("Plane normal must be finite and nonzero.")
        offset_ = float(offset)
        if not np.isfinite(offset_):
            raise ValueError("Plane offset must be finite.")
        features = _analytic_feature_policy(feature_policy)
        self.unit_normal = normal_ / norm
        self.offset = offset_
        self._feature_policy = features
        self._geometry_id = canonical_fingerprint(
            {
                "kind": "plane-contact-geometry",
                "normal": array_tree_fingerprint(np.asarray(self.unit_normal)),
                "dimension": int(normal_.size),
                "offset": offset_.hex(),
                "feature_policy": features.policy_id,
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return int(self.unit_normal.size)

    @property
    def geometry_id(self) -> str:
        return self._geometry_id

    @property
    def feature_policy(self) -> CollisionFeaturePolicy:
        return self._feature_policy

    @property
    def guarantee_level(self) -> ContactGuaranteeLevel:
        return ContactGuaranteeLevel.ANALYTIC_CONSERVATIVE

    def signed_distance(self, points: ArrayLike, /) -> Array:
        points_ = jnp.asarray(points, dtype=self.unit_normal.dtype)
        return jnp.sum(points_ * self.unit_normal, axis=-1) - self.offset

    def normal(self, points: ArrayLike, /) -> Array:
        points_ = jnp.asarray(points, dtype=self.unit_normal.dtype)
        return jnp.broadcast_to(self.unit_normal, points_.shape)

    def bounds(self, /) -> tuple[Array, Array]:
        lower = jnp.full((self.ambient_dimension,), -jnp.inf)
        upper = jnp.full((self.ambient_dimension,), jnp.inf)
        return lower, upper


class FunctionImplicitContactGeometry(AbstractImplicitContactGeometry):
    """Differentiable implicit geometry with explicit guarantee semantics."""

    distance_action: Callable[[Array], Array] = eqx.field(static=True)
    support_action: Callable[[Array], Array] | None = eqx.field(static=True)
    lower_bound: Array
    upper_bound: Array
    _guarantee_level: ContactGuaranteeLevel = eqx.field(static=True)
    _geometry_id: str = eqx.field(static=True)
    _feature_policy: CollisionFeaturePolicy

    def __init__(
        self,
        distance_action: Callable[[Array], Array],
        lower_bound: ArrayLike,
        upper_bound: ArrayLike,
        /,
        *,
        support_action: Callable[[Array], Array] | None = None,
        guarantee_level: ContactGuaranteeLevel = ContactGuaranteeLevel.HEURISTIC,
        feature_policy: CollisionFeaturePolicy,
        geometry_id: str | None = None,
    ):
        if not callable(distance_action):
            raise TypeError("distance_action must be callable.")
        lower = jnp.asarray(lower_bound)
        upper = jnp.asarray(upper_bound, dtype=lower.dtype)
        if lower.shape not in ((2,), (3,)) or upper.shape != lower.shape:
            raise ValueError("Implicit geometry bounds require dimension two or three.")
        if not bool(
            jnp.all(jnp.isfinite(lower) & jnp.isfinite(upper) & (lower <= upper))
        ):
            raise ValueError("Implicit geometry bounds must be finite and ordered.")
        level = ContactGuaranteeLevel(guarantee_level)
        features = _analytic_feature_policy(feature_policy)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "function-implicit-contact-geometry",
                    "dimension": int(lower.size),
                    "distance_action": distance_action,
                    "support_action": support_action,
                    "guarantee": int(level),
                    "feature_policy": features.policy_id,
                }
            )
            if geometry_id is None
            else str(geometry_id)
        )
        if not identifier:
            raise ValueError("geometry_id must be nonempty or None.")
        self.distance_action = distance_action
        self.support_action = support_action
        self.lower_bound = lower
        self.upper_bound = upper
        self._guarantee_level = level
        self._feature_policy = features
        self._geometry_id = identifier

    @property
    def ambient_dimension(self) -> int:
        return int(self.lower_bound.size)

    @property
    def geometry_id(self) -> str:
        return self._geometry_id

    @property
    def feature_policy(self) -> CollisionFeaturePolicy:
        return self._feature_policy

    @property
    def guarantee_level(self) -> ContactGuaranteeLevel:
        return self._guarantee_level

    def signed_distance(self, points: ArrayLike, /) -> Array:
        return jnp.asarray(self.distance_action(jnp.asarray(points)))

    def normal(self, points: ArrayLike, /) -> Array:
        points_ = jnp.asarray(points)
        gradient = jax.vmap(jax.grad(lambda point: self.distance_action(point)))(
            points_.reshape((-1, self.ambient_dimension))
        ).reshape(points_.shape)
        norm = jnp.sqrt(jnp.sum(gradient * gradient, axis=-1, keepdims=True))
        return gradient / jnp.maximum(norm, jnp.finfo(points_.dtype).eps)

    def bounds(self, /) -> tuple[Array, Array]:
        return self.lower_bound, self.upper_bound

    def support(self, directions: ArrayLike, /) -> Array:
        if self.support_action is None:
            raise ValueError("Implicit geometry does not provide a support map.")
        return jnp.asarray(self.support_action(jnp.asarray(directions)))


__all__ = [
    "AbstractImplicitContactGeometry",
    "FunctionImplicitContactGeometry",
    "ImplicitContactEvaluation",
    "PlaneContactGeometry",
    "SphereContactGeometry",
]
