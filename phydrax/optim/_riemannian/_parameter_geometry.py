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
from .._pytree import (
    _parameter_array_leaf_paths,
    _parameter_tree_metadata,
    _validated_parameter_leaves,
)


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

        (
            leaves,
            tree_definition,
            paths,
            shapes,
            dtypes,
        ) = _parameter_tree_metadata(parameters, owner="ParameterGeometry")

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
            zip(paths, leaves, shapes, strict=True)
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
            if manifold.scalar_field == "real":
                valid_dtype = jnp.issubdtype(array.dtype, jnp.floating)
            elif manifold.scalar_field == "complex":
                valid_dtype = jnp.issubdtype(array.dtype, jnp.complexfloating)
            else:
                raise ValueError(
                    f"Unknown manifold scalar field {manifold.scalar_field!r}."
                )
            if not valid_dtype:
                raise TypeError(
                    f"Manifold parameter leaf {path_name} must use "
                    f"{manifold.scalar_field} floating-point coordinates."
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
                dtype=array.real.dtype,
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
        return _parameter_array_leaf_paths(parameters)

    @property
    def num_manifold_leaves(self) -> int:
        return len(self.selected_indices)

    @property
    def manifold_ids(self) -> tuple[str, ...]:
        return tuple(
            manifold.manifold_id for manifold in self.manifolds if manifold is not None
        )

    def _validated_leaves(self, tree: PyTree[Any], name: str, /) -> list[Array]:
        return _validated_parameter_leaves(
            tree,
            tree_definition=self.tree_definition,
            paths=self.paths,
            shapes=self.shapes,
            dtypes=self.dtypes,
            name=name,
        )

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

    def _factor_shape(
        self,
        shape: tuple[int, ...],
        manifold: AbstractRiemannianManifold | None,
        /,
    ) -> tuple[int, ...]:
        if manifold is None:
            return ()
        rank = len(manifold.point_shape)
        return shape[:-rank] if rank else shape

    def _factor_moment_zeros(self, parameters: PyTree[Any], /) -> PyTree[Array]:
        """Return scalar moment storage for each independent geometry factor."""
        points = self._validated_leaves(parameters, "Parameters")
        moments = [
            jnp.zeros(
                self._factor_shape(point.shape, manifold),
                dtype=point.real.dtype,
            )
            for point, manifold in zip(points, self.manifolds, strict=True)
        ]
        return self.tree_definition.unflatten(moments)

    def _factor_squared_norms(
        self,
        parameters: PyTree[Any],
        tangents: PyTree[Any],
        /,
    ) -> PyTree[Array]:
        """Return weighted squared norms without choosing ambient coordinates."""
        points = self._validated_leaves(parameters, "Parameters")
        vectors = self._validated_leaves(tangents, "Tangents")
        outputs: list[Array] = []
        for point, vector, manifold, weight in zip(
            points,
            vectors,
            self.manifolds,
            self.weights,
            strict=True,
        ):
            if manifold is None:
                squared = _real_tree_inner(vector, vector)
            else:
                factor_shape = self._factor_shape(point.shape, manifold)
                if factor_shape:
                    flattened_points = point.reshape((-1,) + manifold.point_shape)
                    flattened_vectors = vector.reshape((-1,) + manifold.point_shape)
                    squared = jax.vmap(
                        lambda point_, vector_: manifold.inner(
                            point_,
                            vector_,
                            vector_,
                        )
                    )(flattened_points, flattened_vectors).reshape(factor_shape)
                else:
                    squared = manifold.inner(point, vector, vector)
            weighted = jnp.asarray(weight, dtype=point.real.dtype) * jnp.real(squared)
            outputs.append(jnp.maximum(weighted, 0.0))
        return self.tree_definition.unflatten(outputs)

    def _scale_tangent_factors(
        self,
        tangent: PyTree[Any],
        factors: PyTree[Any],
        /,
    ) -> PyTree[Array]:
        """Scale each tangent by one broadcastable intrinsic-factor scalar."""
        vectors = self._validated_leaves(tangent, "Tangent")
        factor_leaves, factor_definition = jax.tree_util.tree_flatten(factors)
        if factor_definition != self.tree_definition:
            raise ValueError("Factors have an incompatible PyTree structure.")
        outputs: list[Array] = []
        for path, vector, factor, shape, manifold in zip(
            self.paths,
            vectors,
            factor_leaves,
            self.shapes,
            self.manifolds,
            strict=True,
        ):
            factor_array = jnp.asarray(factor)
            expected = self._factor_shape(shape, manifold)
            if factor_array.shape != expected:
                raise ValueError(
                    f"Factor leaf {path} must have shape {expected}, "
                    f"got {factor_array.shape}."
                )
            rank = 0 if manifold is None else len(manifold.point_shape)
            broadcast = factor_array.reshape(expected + (1,) * rank)
            outputs.append(vector * broadcast)
        return self.tree_definition.unflatten(outputs)

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
