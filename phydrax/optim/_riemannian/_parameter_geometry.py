#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from ..._strict import StrictModule
from ...metrix import AbstractRiemannianManifold


def _path_string(path: tuple[Any, ...], /) -> str:
    return jax.tree_util.keystr(path) or "<root>"


def _real_tree_inner(left: Array, right: Array, /) -> Array:
    return jnp.real(jnp.vdot(left, right))


class ParameterGeometry(StrictModule):
    """Bind selected inexact-array PyTree leaves to declared manifolds.

    The binding records the exact trainable PyTree structure, shape, and dtype. A
    selected leaf may have leading product axes before the manifold's trailing
    ``point_shape``. Unselected leaves retain ordinary Euclidean geometry.
    """

    tree_definition: Any = eqx.field(static=True)
    paths: tuple[str, ...] = eqx.field(static=True)
    shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    dtypes: tuple[str, ...] = eqx.field(static=True)
    manifolds: tuple[AbstractRiemannianManifold | None, ...]
    weights: tuple[float, ...] = eqx.field(static=True)
    selected_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        parameters: PyTree[Any],
        manifolds: Mapping[str, AbstractRiemannianManifold],
        /,
        *,
        weights: Mapping[str, ArrayLike] | None = None,
    ):
        if not isinstance(manifolds, Mapping):
            raise TypeError("ParameterGeometry manifolds must be a path mapping.")
        if not manifolds:
            raise ValueError("ParameterGeometry requires at least one manifold leaf.")
        if weights is not None and not isinstance(weights, Mapping):
            raise TypeError("ParameterGeometry weights must be a path mapping.")

        path_leaves, tree_definition = jax.tree_util.tree_flatten_with_path(parameters)
        if not path_leaves:
            raise ValueError("ParameterGeometry requires at least one trainable array.")

        paths: list[str] = []
        shapes: list[tuple[int, ...]] = []
        dtypes: list[str] = []
        for path, leaf in path_leaves:
            path_name = _path_string(path)
            if not eqx.is_inexact_array(leaf):
                raise TypeError(
                    "ParameterGeometry requires a filtered trainable PyTree containing "
                    f"only inexact arrays; leaf {path_name} is {type(leaf).__name__}."
                )
            array = jnp.asarray(leaf)
            paths.append(path_name)
            shapes.append(tuple(int(size) for size in array.shape))
            dtypes.append(str(array.dtype))

        available = set(paths)
        requested = set(manifolds)
        missing = tuple(sorted(requested - available))
        if missing:
            raise ValueError(
                f"Unknown ParameterGeometry leaf paths {missing}; available paths are "
                f"{tuple(paths)}."
            )
        requested_weights = {} if weights is None else dict(weights)
        unknown_weights = tuple(sorted(set(requested_weights) - requested))
        if unknown_weights:
            raise ValueError(
                "ParameterGeometry weights may select only manifold-bound leaves; "
                f"got {unknown_weights}."
            )

        leaf_manifolds: list[AbstractRiemannianManifold | None] = []
        selected_indices: list[int] = []
        leaf_weights: list[float] = []
        for index, (path_name, leaf, shape) in enumerate(
            zip(paths, (leaf for _, leaf in path_leaves), shapes, strict=True)
        ):
            manifold = manifolds[path_name] if path_name in manifolds else None
            if manifold is None:
                leaf_manifolds.append(None)
                leaf_weights.append(1.0)
                continue
            if not isinstance(manifold, AbstractRiemannianManifold):
                raise TypeError(
                    f"ParameterGeometry leaf {path_name} must be bound to an "
                    "AbstractRiemannianManifold."
                )
            array = jnp.asarray(leaf)
            if not jnp.issubdtype(array.dtype, jnp.floating):
                raise TypeError(
                    f"Manifold parameter leaf {path_name} must be real floating-point."
                )
            point_shape = manifold.point_shape
            rank = len(point_shape)
            if array.ndim < rank or (rank and shape[-rank:] != point_shape):
                raise ValueError(
                    f"Manifold parameter leaf {path_name} must have trailing shape "
                    f"{point_shape}, got {shape}."
                )
            membership = jnp.asarray(manifold.contains(array), dtype=bool)
            if membership.shape != ():
                raise ValueError(
                    f"Manifold {manifold.manifold_id} contains() must return a scalar."
                )
            if not bool(membership):
                raise ValueError(
                    f"Initial parameter leaf {path_name} is outside "
                    f"{manifold.manifold_id}."
                )
            leaf_manifolds.append(manifold)
            selected_indices.append(index)
            weight = jnp.asarray(
                requested_weights.get(path_name, 1.0),
                dtype=array.dtype,
            )
            if weight.shape != ():
                raise ValueError(
                    f"ParameterGeometry weight for {path_name} must be scalar."
                )
            if not bool(jax.device_get(jnp.isfinite(weight) & (weight > 0.0))):
                raise ValueError(
                    f"ParameterGeometry weight for {path_name} must be finite and positive."
                )
            leaf_weights.append(float(jax.device_get(weight)))

        self.tree_definition = tree_definition
        self.paths = tuple(paths)
        self.shapes = tuple(shapes)
        self.dtypes = tuple(dtypes)
        self.manifolds = tuple(leaf_manifolds)
        self.selected_indices = tuple(selected_indices)
        self.weights = tuple(leaf_weights)

    @classmethod
    def from_leaf_paths(
        cls,
        parameters: PyTree[Any],
        manifolds: Mapping[str, AbstractRiemannianManifold],
        /,
        *,
        weights: Mapping[str, ArrayLike] | None = None,
    ) -> "ParameterGeometry":
        """Construct a binding from deterministic ``jax.tree_util.keystr`` paths."""
        return cls(parameters, manifolds, weights=weights)

    @staticmethod
    def array_leaf_paths(parameters: PyTree[Any], /) -> tuple[str, ...]:
        """Return deterministic paths for all inexact-array leaves."""
        return tuple(
            _path_string(path)
            for path, leaf in jax.tree_util.tree_flatten_with_path(parameters)[0]
            if eqx.is_inexact_array(leaf)
        )

    @property
    def num_manifold_leaves(self) -> int:
        return len(self.selected_indices)

    @property
    def manifold_ids(self) -> tuple[str, ...]:
        return tuple(
            manifold.manifold_id for manifold in self.manifolds if manifold is not None
        )

    def _validated_leaves(self, tree: PyTree[Any], name: str, /) -> list[Array]:
        leaves, tree_definition = jax.tree_util.tree_flatten(tree)
        if tree_definition != self.tree_definition:
            raise ValueError(f"{name} has an incompatible PyTree structure.")
        validated: list[Array] = []
        for path, leaf, shape, dtype in zip(
            self.paths,
            leaves,
            self.shapes,
            self.dtypes,
            strict=True,
        ):
            if not eqx.is_inexact_array(leaf):
                raise TypeError(f"{name} leaf {path} must be an inexact array.")
            array = jnp.asarray(leaf)
            if array.shape != shape:
                raise ValueError(
                    f"{name} leaf {path} must have shape {shape}, got {array.shape}."
                )
            if str(array.dtype) != dtype:
                raise TypeError(
                    f"{name} leaf {path} must have dtype {dtype}, got {array.dtype}."
                )
            validated.append(array)
        return validated

    def validate(self, parameters: PyTree[Any], /) -> None:
        """Validate the bound structure, shapes, and dtypes."""
        self._validated_leaves(parameters, "Parameters")

    def egrad_to_rgrad(
        self,
        parameters: PyTree[Any],
        gradients: PyTree[Any],
        /,
    ) -> PyTree[Array]:
        points = self._validated_leaves(parameters, "Parameters")
        cotangents = self._validated_leaves(gradients, "Gradients")
        outputs = [
            cotangent
            if manifold is None
            else manifold.egrad_to_rgrad(point, cotangent) / weight
            for point, cotangent, manifold, weight in zip(
                points, cotangents, self.manifolds, self.weights, strict=True
            )
        ]
        return self.tree_definition.unflatten(outputs)

    def inner(
        self,
        parameters: PyTree[Any],
        left_tangent: PyTree[Any],
        right_tangent: PyTree[Any],
        /,
    ) -> Array:
        points = self._validated_leaves(parameters, "Parameters")
        left = self._validated_leaves(left_tangent, "Left tangent")
        right = self._validated_leaves(right_tangent, "Right tangent")
        value = jnp.asarray(0.0)
        for point, left_leaf, right_leaf, manifold, weight in zip(
            points, left, right, self.manifolds, self.weights, strict=True
        ):
            term = (
                _real_tree_inner(left_leaf, right_leaf)
                if manifold is None
                else manifold.inner(point, left_leaf, right_leaf)
            )
            value = value + weight * jnp.asarray(term).reshape(())
        return jnp.real(value)

    def norm(self, parameters: PyTree[Any], tangent: PyTree[Any], /) -> Array:
        """Return the norm induced by the product of declared leaf metrics."""
        squared = self.inner(parameters, tangent, tangent)
        return jnp.sqrt(jnp.maximum(squared, 0.0))

    def project_tangent(
        self,
        parameters: PyTree[Any],
        ambient_vectors: PyTree[Any],
        /,
    ) -> PyTree[Array]:
        """Project selected leaves and preserve Euclidean leaves."""
        points = self._validated_leaves(parameters, "Parameters")
        vectors = self._validated_leaves(ambient_vectors, "Ambient vectors")
        outputs = [
            vector if manifold is None else manifold.project_tangent(point, vector)
            for point, vector, manifold in zip(
                points, vectors, self.manifolds, strict=True
            )
        ]
        return self.tree_definition.unflatten(outputs)

    def maximum_tangent_residual(
        self,
        parameters: PyTree[Any],
        tangents: PyTree[Any],
        /,
    ) -> Array:
        """Return the largest ambient residual from tangent projection."""
        points = self._validated_leaves(parameters, "Parameters")
        vectors = self._validated_leaves(tangents, "Tangents")
        residual = jnp.asarray(0.0)
        for point, vector, manifold in zip(points, vectors, self.manifolds, strict=True):
            if manifold is not None:
                projection = manifold.project_tangent(point, vector)
                residual = jnp.maximum(
                    residual,
                    jnp.linalg.norm((vector - projection).reshape((-1,))),
                )
        return residual

    def retract(
        self,
        parameters: PyTree[Any],
        tangent_steps: PyTree[Any],
        /,
    ) -> PyTree[Array]:
        points = self._validated_leaves(parameters, "Parameters")
        steps = self._validated_leaves(tangent_steps, "Tangent steps")
        outputs = [
            point + step if manifold is None else manifold.retract(point, step)
            for point, step, manifold in zip(points, steps, self.manifolds, strict=True)
        ]
        return self.tree_definition.unflatten(outputs)

    def transport(
        self,
        parameters: PyTree[Any],
        tangent_steps: PyTree[Any],
        destinations: PyTree[Any],
        tangents: PyTree[Any],
        /,
    ) -> PyTree[Array]:
        points = self._validated_leaves(parameters, "Parameters")
        steps = self._validated_leaves(tangent_steps, "Tangent steps")
        targets = self._validated_leaves(destinations, "Destinations")
        vectors = self._validated_leaves(tangents, "Tangents")
        outputs = [
            vector
            if manifold is None
            else manifold.transport(point, step, target, vector)
            for point, step, target, vector, manifold in zip(
                points, steps, targets, vectors, self.manifolds, strict=True
            )
        ]
        return self.tree_definition.unflatten(outputs)

    def maximum_transport_metric_distortion(
        self,
        parameters: PyTree[Any],
        destinations: PyTree[Any],
        source_tangents: PyTree[Any],
        transported_tangents: PyTree[Any],
        /,
    ) -> Array:
        """Return the largest relative norm defect for claimed isometric transports."""
        points = self._validated_leaves(parameters, "Parameters")
        targets = self._validated_leaves(destinations, "Destinations")
        source = self._validated_leaves(source_tangents, "Source tangents")
        transported = self._validated_leaves(transported_tangents, "Transported tangents")
        residual = jnp.asarray(0.0)
        for point, target, before, after, manifold in zip(
            points,
            targets,
            source,
            transported,
            self.manifolds,
            strict=True,
        ):
            if manifold is not None and manifold.transport_is_isometric:
                source_squared = manifold.inner(point, before, before)
                target_squared = manifold.inner(target, after, after)
                scale = jnp.maximum(jnp.abs(source_squared), 1.0)
                residual = jnp.maximum(
                    residual,
                    jnp.abs(target_squared - source_squared) / scale,
                )
        return residual

    def contains(self, parameters: PyTree[Any], /) -> Array:
        points = self._validated_leaves(parameters, "Parameters")
        membership = jnp.asarray(True)
        for point, manifold in zip(points, self.manifolds, strict=True):
            leaf_membership = (
                jnp.all(jnp.isfinite(point))
                if manifold is None
                else manifold.contains(point)
            )
            membership = membership & jnp.asarray(leaf_membership, dtype=bool).reshape(())
        return membership

    def constraint_residuals(
        self,
        parameters: PyTree[Any],
        /,
    ) -> dict[str, Array]:
        points = self._validated_leaves(parameters, "Parameters")
        return {
            path: jnp.asarray(manifold.constraint_residual(point)).reshape(())
            for path, point, manifold in zip(
                self.paths, points, self.manifolds, strict=True
            )
            if manifold is not None
        }

    def maximum_constraint_residual(self, parameters: PyTree[Any], /) -> Array:
        points = self._validated_leaves(parameters, "Parameters")
        residual = jnp.asarray(0.0)
        for point, manifold in zip(points, self.manifolds, strict=True):
            if manifold is not None:
                residual = jnp.maximum(
                    residual,
                    jnp.asarray(manifold.constraint_residual(point)).reshape(()),
                )
        return residual


__all__ = ["ParameterGeometry"]
