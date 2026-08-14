#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._spaces import _coordinate_dtype, AbstractVectorSpace


CompatibilityMode = Literal["error", "project"]
GaugeMode = Literal["minimum-norm", "project"]


def _basis_gram(
    space: AbstractVectorSpace,
    basis: Array,
    dimension: Array,
    /,
) -> tuple[Array, Array]:
    mask = jnp.arange(basis.shape[1]) < dimension
    active_basis = jnp.where(mask[None, :], basis, 0)

    def inner(left, right):
        return space.inner(space.unflatten(left), space.unflatten(right))

    gram = jax.vmap(
        lambda left: jax.vmap(lambda right: inner(left, right), in_axes=1)(active_basis),
        in_axes=1,
    )(active_basis)
    gram = gram + jnp.diag((~mask).astype(gram.dtype))
    return gram, active_basis


class LinearSubspace(StrictModule):
    """Fixed-capacity coordinate basis with a dynamic effective dimension."""

    space: AbstractVectorSpace
    basis: Array
    dimension: Array
    orthonormal: bool = eqx.field(static=True)
    subspace_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: AbstractVectorSpace,
        basis: ArrayLike,
        /,
        *,
        dimension: int | ArrayLike | None = None,
        orthonormal: bool = False,
        subspace_id: str | None = None,
    ):
        if not isinstance(space, AbstractVectorSpace):
            raise TypeError("space must be an AbstractVectorSpace.")
        basis_ = jnp.asarray(basis)
        if basis_.ndim != 2 or basis_.shape[0] != space.size:
            raise ValueError("Subspace basis must have shape (space.size, capacity).")
        if not jnp.issubdtype(basis_.dtype, jnp.inexact):
            raise TypeError("Subspace basis must use an inexact dtype.")
        if basis_.dtype != _coordinate_dtype(space):
            raise TypeError("Subspace basis dtype must match its coordinate space.")
        capacity = int(basis_.shape[1])
        dimension_ = jnp.asarray(
            capacity if dimension is None else dimension,
            dtype=jnp.int32,
        )
        if dimension_.shape != ():
            raise ValueError("Subspace dimension must be scalar.")
        dimension_ = eqx.error_if(
            dimension_,
            (dimension_ < 0) | (dimension_ > capacity),
            "Subspace dimension must lie between zero and basis capacity.",
        )
        basis_ = eqx.error_if(
            basis_,
            jnp.any(~jnp.isfinite(basis_)),
            "Subspace basis entries must be finite.",
        )
        if capacity:
            gram, active_basis = _basis_gram(space, basis_, dimension_)
            norms = jnp.sqrt(jnp.maximum(jnp.real(jnp.diag(gram)), 0.0))
            mask = jnp.arange(capacity) < dimension_
            scales = jnp.where(mask & (norms > 0.0), norms, 1.0)
            normalized_gram, _ = _basis_gram(
                space,
                active_basis / scales[None, :],
                dimension_,
            )
            singular_values = jnp.linalg.svd(
                normalized_gram,
                compute_uv=False,
            )
            cutoff = (
                jnp.finfo(basis_.real.dtype).eps
                * max(capacity, 1)
                * jnp.max(singular_values)
            )
            basis_ = eqx.error_if(
                basis_,
                jnp.any(~jnp.isfinite(singular_values))
                | (jnp.min(singular_values) <= cutoff),
                "Active subspace basis columns must be linearly independent.",
            )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "linear-subspace",
                    "space": space.space_id,
                    "capacity": capacity,
                    "orthonormal": bool(orthonormal),
                }
            )
            if subspace_id is None
            else str(subspace_id)
        )
        if not identifier:
            raise ValueError("subspace_id must be non-empty.")
        self.space = space
        self.basis = basis_
        self.dimension = dimension_
        self.orthonormal = bool(orthonormal)
        self.subspace_id = identifier

    @property
    def capacity(self) -> int:
        return int(self.basis.shape[1])

    def project_coordinates(self, coordinates: ArrayLike, /) -> Array:
        value = jnp.asarray(coordinates)
        if value.shape != (self.space.size,):
            raise ValueError("Projection coordinates must match the subspace space.")
        if value.dtype != _coordinate_dtype(self.space):
            raise TypeError("Projection coordinate dtype must match the subspace space.")
        if self.capacity == 0:
            return jnp.zeros_like(value)
        gram, basis = _basis_gram(self.space, self.basis, self.dimension)

        def inner(column):
            return self.space.inner(
                self.space.unflatten(column),
                self.space.unflatten(value),
            )

        right_hand_side = jax.vmap(inner, in_axes=1)(basis)
        coefficients = jnp.linalg.solve(gram, right_hand_side)
        return basis @ coefficients

    def project(self, vector: PyTree[Any], /) -> PyTree[Array]:
        coordinates = self.space.flatten(self.space.validate(vector))
        return self.space.unflatten(self.project_coordinates(coordinates))

    def orthogonal_component(self, vector: PyTree[Any], /) -> PyTree[Array]:
        value = self.space.validate(vector)
        projected = self.project(value)
        return jax.tree.map(lambda left, right: left - right, value, projected)

    def projection_norm(self, vector: PyTree[Any], /) -> Array:
        projected = self.project(vector)
        return jnp.sqrt(jnp.real(self.space.inner(projected, projected)))


class NullspacePolicy(StrictModule):
    """Known nullspaces plus explicit compatibility and gauge behavior."""

    right: LinearSubspace | None
    left: LinearSubspace | None
    compatibility: CompatibilityMode = eqx.field(static=True)
    gauge: GaugeMode = eqx.field(static=True)

    def __init__(
        self,
        *,
        right: LinearSubspace | None = None,
        left: LinearSubspace | None = None,
        compatibility: CompatibilityMode = "error",
        gauge: GaugeMode = "minimum-norm",
    ):
        if right is not None and not isinstance(right, LinearSubspace):
            raise TypeError("right must be a LinearSubspace or None.")
        if left is not None and not isinstance(left, LinearSubspace):
            raise TypeError("left must be a LinearSubspace or None.")
        if compatibility not in ("error", "project"):
            raise ValueError("compatibility must be 'error' or 'project'.")
        if gauge not in ("minimum-norm", "project"):
            raise ValueError("gauge must be 'minimum-norm' or 'project'.")
        self.right = right
        self.left = left
        self.compatibility = compatibility
        self.gauge = gauge


__all__ = [
    "CompatibilityMode",
    "GaugeMode",
    "LinearSubspace",
    "NullspacePolicy",
]
