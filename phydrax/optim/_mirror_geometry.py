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

from .._strict import StrictModule
from ..metrix import LegendreGeometry
from ._pytree import (
    _parameter_array_leaf_paths,
    _parameter_tree_metadata,
    _validated_parameter_leaves,
)


class ParameterMirrorGeometry(StrictModule):
    """Bind selected real PyTree leaves to separable Legendre geometries."""

    tree_definition: Any = eqx.field(static=True)
    paths: tuple[str, ...] = eqx.field(static=True)
    shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    dtypes: tuple[str, ...] = eqx.field(static=True)
    geometries: tuple[LegendreGeometry | None, ...]
    weights: tuple[float, ...] = eqx.field(static=True)
    selected_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        parameters: PyTree[Any],
        geometries: Mapping[str, LegendreGeometry],
        /,
        *,
        weights: Mapping[str, ArrayLike] | None = None,
    ):
        if not isinstance(geometries, Mapping):
            raise TypeError("ParameterMirrorGeometry geometries must be a path mapping.")
        if not geometries:
            raise ValueError(
                "ParameterMirrorGeometry requires at least one Legendre leaf."
            )
        if weights is not None and not isinstance(weights, Mapping):
            raise TypeError("ParameterMirrorGeometry weights must be a path mapping.")

        (
            leaves,
            tree_definition,
            paths,
            shapes,
            dtypes,
        ) = _parameter_tree_metadata(parameters, owner="ParameterMirrorGeometry")
        available = set(paths)
        requested = set(geometries)
        missing = tuple(sorted(requested - available))
        if missing:
            raise ValueError(
                f"Unknown ParameterMirrorGeometry leaf paths {missing}; available "
                f"paths are {paths}."
            )
        requested_weights = {} if weights is None else dict(weights)
        unknown_weights = tuple(sorted(set(requested_weights) - requested))
        if unknown_weights:
            raise ValueError(
                "ParameterMirrorGeometry weights may select only Legendre-bound "
                f"leaves; got {unknown_weights}."
            )

        leaf_geometries: list[LegendreGeometry | None] = []
        selected_indices: list[int] = []
        leaf_weights: list[float] = []
        for index, (path, point, shape) in enumerate(
            zip(paths, leaves, shapes, strict=True)
        ):
            if path not in geometries:
                leaf_geometries.append(None)
                leaf_weights.append(1.0)
                continue
            geometry = geometries[path]
            if not isinstance(geometry, LegendreGeometry):
                raise TypeError(
                    f"ParameterMirrorGeometry leaf {path} must be bound to a "
                    "LegendreGeometry."
                )
            if not jnp.issubdtype(point.dtype, jnp.floating):
                raise TypeError(
                    f"Mirror parameter leaf {path} must use real floating-point "
                    "coordinates."
                )
            if point.ndim < 1 or shape[-1] != geometry.dimension:
                raise ValueError(
                    f"Mirror parameter leaf {path} must have trailing dimension "
                    f"{geometry.dimension}, got {shape}."
                )
            membership = jnp.asarray(geometry.primal_contains(point), dtype=bool)
            if membership.shape != point.shape[:-1]:
                raise ValueError(
                    f"Legendre geometry {geometry.geometry_id!r} membership must "
                    "preserve leading parameter axes."
                )
            if not bool(jax.device_get(jnp.all(membership))):
                raise ValueError(
                    f"Initial parameter leaf {path} is outside "
                    f"{geometry.geometry_id!r}."
                )
            weight = jnp.asarray(
                requested_weights.get(path, 1.0),
                dtype=point.dtype,
            )
            if weight.shape != ():
                raise ValueError(
                    f"ParameterMirrorGeometry weight for {path} must be scalar."
                )
            if not bool(jax.device_get(jnp.isfinite(weight) & (weight > 0.0))):
                raise ValueError(
                    f"ParameterMirrorGeometry weight for {path} must be finite "
                    "and positive."
                )
            leaf_geometries.append(geometry)
            selected_indices.append(index)
            leaf_weights.append(float(jax.device_get(weight)))

        self.tree_definition = tree_definition
        self.paths = paths
        self.shapes = shapes
        self.dtypes = dtypes
        self.geometries = tuple(leaf_geometries)
        self.weights = tuple(leaf_weights)
        self.selected_indices = tuple(selected_indices)

    @classmethod
    def from_leaf_paths(
        cls,
        parameters: PyTree[Any],
        geometries: Mapping[str, LegendreGeometry],
        /,
        *,
        weights: Mapping[str, ArrayLike] | None = None,
    ) -> ParameterMirrorGeometry:
        """Construct a binding from deterministic ``jax.tree_util.keystr`` paths."""
        return cls(parameters, geometries, weights=weights)

    @staticmethod
    def array_leaf_paths(parameters: PyTree[Any], /) -> tuple[str, ...]:
        """Return deterministic paths for all inexact-array leaves."""
        return _parameter_array_leaf_paths(parameters)

    @property
    def num_legendre_leaves(self) -> int:
        return len(self.selected_indices)

    @property
    def geometry_ids(self) -> tuple[str, ...]:
        return tuple(
            geometry.geometry_id
            for geometry in self.geometries
            if geometry is not None
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

    def contains(self, parameters: PyTree[Any], /) -> Array:
        points = self._validated_leaves(parameters, "Parameters")
        membership = jnp.asarray(True)
        for point, geometry in zip(points, self.geometries, strict=True):
            leaf_membership = (
                jnp.all(jnp.isfinite(point))
                if geometry is None
                else jnp.all(geometry.primal_contains(point))
            )
            membership = membership & jnp.asarray(leaf_membership, dtype=bool)
        return membership

    def constraint_residuals(
        self,
        parameters: PyTree[Any],
        /,
    ) -> dict[str, Array]:
        points = self._validated_leaves(parameters, "Parameters")
        return {
            path: jnp.where(
                jnp.all(geometry.primal_contains(point)),
                jnp.asarray(0.0, dtype=point.dtype),
                jnp.asarray(jnp.inf, dtype=point.dtype),
            )
            for path, point, geometry in zip(
                self.paths,
                points,
                self.geometries,
                strict=True,
            )
            if geometry is not None
        }

    def maximum_constraint_residual(self, parameters: PyTree[Any], /) -> Array:
        residuals = self.constraint_residuals(parameters)
        residual = jnp.asarray(0.0)
        for value in residuals.values():
            residual = jnp.maximum(residual, value)
        return residual

    def dual_translate(
        self,
        parameters: PyTree[Any],
        dual_displacements: PyTree[Any],
        /,
    ) -> PyTree[Array]:
        points = self._validated_leaves(parameters, "Parameters")
        displacements = self._validated_leaves(
            dual_displacements,
            "Dual displacements",
        )
        destinations: list[Array] = []
        for path, point, displacement, geometry, weight in zip(
            self.paths,
            points,
            displacements,
            self.geometries,
            self.weights,
            strict=True,
        ):
            scaled = displacement / jnp.asarray(weight, dtype=point.dtype)
            if geometry is None:
                destination = point + scaled
                destination = eqx.error_if(
                    destination,
                    jnp.any(~jnp.isfinite(destination)),
                    f"Euclidean mirror leaf {path} produced nonfinite coordinates.",
                )
            else:
                destination = geometry.dual_translate(point, scaled)
            destinations.append(destination)
        return self.tree_definition.unflatten(destinations)

    def coordinate_gradient_norm(self, gradients: PyTree[Any], /) -> Array:
        leaves = self._validated_leaves(gradients, "Gradients")
        squared = sum(
            (jnp.real(jnp.vdot(leaf, leaf)) for leaf in leaves),
            start=jnp.asarray(0.0),
        )
        return jnp.sqrt(jnp.maximum(squared, 0.0))

    def dual_displacement_norm(
        self,
        dual_displacements: PyTree[Any],
        /,
    ) -> Array:
        displacements = self._validated_leaves(
            dual_displacements,
            "Dual displacements",
        )
        squared = jnp.asarray(0.0)
        for displacement, weight in zip(
            displacements,
            self.weights,
            strict=True,
        ):
            scaled = displacement / jnp.asarray(weight, dtype=displacement.dtype)
            squared = squared + jnp.real(jnp.vdot(scaled, scaled))
        return jnp.sqrt(jnp.maximum(squared, 0.0))

    def bregman_step(
        self,
        previous: PyTree[Any],
        destination: PyTree[Any],
        /,
    ) -> Array:
        left = self._validated_leaves(destination, "Destination parameters")
        right = self._validated_leaves(previous, "Previous parameters")
        divergence = jnp.asarray(0.0)
        for left_leaf, right_leaf, geometry, weight in zip(
            left,
            right,
            self.geometries,
            self.weights,
            strict=True,
        ):
            if geometry is None:
                value = 0.5 * jnp.real(
                    jnp.vdot(left_leaf - right_leaf, left_leaf - right_leaf)
                )
            else:
                value = jnp.sum(
                    geometry.bregman_divergence(left_leaf, right_leaf)
                )
            divergence = divergence + jnp.asarray(weight, dtype=value.dtype) * value
        return jnp.real(divergence)


__all__ = ["ParameterMirrorGeometry"]
