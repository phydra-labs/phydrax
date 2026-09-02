#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._certificates import KernelCertificate
from ._spaces import _coordinate_dtype, AbstractVectorSpace


CompatibilityMode = Literal["error", "project"]
GaugeMode = Literal["minimum-norm", "project"]


def _basis_gram(
    space: AbstractVectorSpace,
    basis: Array,
    dimension: Array,
    /,
) -> tuple[Array, Array]:
    batch_shape = basis.shape[:-2]
    capacity = basis.shape[-1]
    mask = jnp.arange(capacity) < dimension[..., None]
    active_basis = jnp.where(mask[..., None, :], basis, 0)
    batch_count = math.prod(batch_shape) if batch_shape else 1
    flattened = active_basis.reshape((batch_count, space.size, capacity))

    def gram_one(columns):
        def inner(left, right):
            return space.inner(space.unflatten(left), space.unflatten(right))

        return jax.vmap(
            lambda left: jax.vmap(lambda right: inner(left, right), in_axes=1)(columns),
            in_axes=1,
        )(columns)

    gram = jax.vmap(gram_one)(flattened).reshape(batch_shape + (capacity, capacity))
    gram = gram + jnp.eye(capacity, dtype=gram.dtype) * (~mask)[..., None, :]
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
        if basis_.ndim < 2 or basis_.shape[-2] != space.size:
            raise ValueError(
                "Subspace basis must have shape batch_shape + (space.size, capacity)."
            )
        if not jnp.issubdtype(basis_.dtype, jnp.inexact):
            raise TypeError("Subspace basis must use an inexact dtype.")
        if basis_.dtype != _coordinate_dtype(space):
            raise TypeError("Subspace basis dtype must match its coordinate space.")
        batch_shape = tuple(int(size) for size in basis_.shape[:-2])
        capacity = int(basis_.shape[-1])
        dimension_ = jnp.asarray(
            jnp.full(batch_shape, capacity, dtype=jnp.int32)
            if dimension is None
            else dimension,
            dtype=jnp.int32,
        )
        if dimension_.shape != batch_shape:
            raise ValueError("Subspace dimension must have shape batch_shape.")
        dimension_ = eqx.error_if(
            dimension_,
            jnp.any((dimension_ < 0) | (dimension_ > capacity)),
            "Subspace dimension must lie between zero and basis capacity.",
        )
        basis_ = eqx.error_if(
            basis_,
            jnp.any(~jnp.isfinite(basis_)),
            "Subspace basis entries must be finite.",
        )
        if capacity:
            gram, active_basis = _basis_gram(space, basis_, dimension_)
            norms = jnp.sqrt(
                jnp.maximum(jnp.real(jnp.diagonal(gram, axis1=-2, axis2=-1)), 0.0)
            )
            mask = jnp.arange(capacity) < dimension_[..., None]
            scales = jnp.where(mask & (norms > 0.0), norms, 1.0)
            normalized_gram, _ = _basis_gram(
                space,
                active_basis / scales[..., None, :],
                dimension_,
            )
            singular_values = jnp.linalg.svd(
                normalized_gram,
                compute_uv=False,
            )
            cutoff = (
                jnp.finfo(basis_.real.dtype).eps
                * max(capacity, 1)
                * jnp.max(singular_values, axis=-1)
            )
            basis_ = eqx.error_if(
                basis_,
                jnp.any(~jnp.isfinite(singular_values))
                | jnp.any(jnp.min(singular_values, axis=-1) <= cutoff),
                "Active subspace basis columns must be linearly independent.",
            )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "linear-subspace",
                    "space": space.space_id,
                    "batch_shape": list(batch_shape),
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
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.basis.shape[:-2])

    @property
    def capacity(self) -> int:
        return int(self.basis.shape[-1])

    def project_coordinates(self, coordinates: ArrayLike, /) -> Array:
        value = jnp.asarray(coordinates)
        shared_shape = (self.space.size,)
        batched_shape = self.batch_shape + shared_shape
        if value.shape == shared_shape:
            value = jnp.broadcast_to(value, batched_shape)
        elif value.shape != batched_shape:
            raise ValueError(
                "Projection coordinates must be shared or match batch_shape + "
                "(space.size,)."
            )
        if value.dtype != _coordinate_dtype(self.space):
            raise TypeError("Projection coordinate dtype must match the subspace space.")
        if self.capacity == 0:
            return jnp.zeros_like(value)
        gram, basis = _basis_gram(self.space, self.basis, self.dimension)
        batch_count = math.prod(self.batch_shape) if self.batch_shape else 1
        basis_flat = basis.reshape((batch_count, self.space.size, self.capacity))
        gram_flat = gram.reshape((batch_count, self.capacity, self.capacity))
        value_flat = value.reshape((batch_count, self.space.size))

        def project_one(columns, matrix, vector):
            def inner(column):
                return self.space.inner(
                    self.space.unflatten(column),
                    self.space.unflatten(vector),
                )

            right_hand_side = jax.vmap(inner, in_axes=1)(columns)
            coefficients = jnp.linalg.solve(matrix, right_hand_side)
            return columns @ coefficients

        projected = jax.vmap(project_one)(basis_flat, gram_flat, value_flat)
        return projected.reshape(batched_shape)

    def project(self, vector: PyTree[Any], /) -> PyTree[Array]:
        coordinates = _flatten_batched(self.space, vector, self.batch_shape)
        return _unflatten_batched(
            self.space,
            self.project_coordinates(coordinates),
            self.batch_shape,
        )

    def orthogonal_component(self, vector: PyTree[Any], /) -> PyTree[Array]:
        value = _broadcast_batched(self.space, vector, self.batch_shape)
        projected = self.project(value)
        return jax.tree.map(lambda left, right: left - right, value, projected)

    def projection_norm(self, vector: PyTree[Any], /) -> Array:
        coordinates = self.project_coordinates(
            _flatten_batched(self.space, vector, self.batch_shape)
        )
        batch_count = math.prod(self.batch_shape) if self.batch_shape else 1
        flattened = coordinates.reshape((batch_count, self.space.size))
        norms = jax.vmap(
            lambda value: jnp.sqrt(
                jnp.real(
                    self.space.inner(
                        self.space.unflatten(value),
                        self.space.unflatten(value),
                    )
                )
            )
        )(flattened)
        return norms.reshape(self.batch_shape)


def _broadcast_batched(
    space: AbstractVectorSpace,
    vector: PyTree[Any],
    batch_shape: tuple[int, ...],
    /,
) -> PyTree[Array]:
    leaves, treedef = jax.tree.flatten(vector)
    specs, expected_treedef = jax.tree.flatten(space.structure())
    if treedef != expected_treedef:
        raise ValueError("Vector PyTree structure does not match the vector space.")
    arrays = []
    for leaf, spec in zip(leaves, specs, strict=True):
        value = jnp.asarray(leaf)
        if value.shape == spec.shape:
            value = jnp.broadcast_to(value, batch_shape + spec.shape)
        elif value.shape != batch_shape + spec.shape:
            raise ValueError("Vector leaves must be shared or carry subspace batch axes.")
        if value.dtype != spec.dtype:
            raise TypeError("Vector leaf dtype must match the subspace space.")
        arrays.append(value)
    return jax.tree.unflatten(treedef, arrays)


def _flatten_batched(
    space: AbstractVectorSpace,
    vector: PyTree[Any],
    batch_shape: tuple[int, ...],
    /,
) -> Array:
    batched = _broadcast_batched(space, vector, batch_shape)
    leaves = jax.tree.leaves(batched)
    flattened = tuple(leaf.reshape(batch_shape + (-1,)) for leaf in leaves)
    return flattened[0] if len(flattened) == 1 else jnp.concatenate(flattened, axis=-1)


def _unflatten_batched(
    space: AbstractVectorSpace,
    coordinates: Array,
    batch_shape: tuple[int, ...],
    /,
) -> PyTree[Array]:
    value = jnp.asarray(coordinates)
    if value.shape != batch_shape + (space.size,):
        raise ValueError("Batched coordinates do not match the vector space.")
    specs, treedef = jax.tree.flatten(space.structure())
    leaves = []
    offset = 0
    for spec in specs:
        count = math.prod(spec.shape)
        leaves.append(
            value[..., offset : offset + count].reshape(batch_shape + spec.shape)
        )
        offset += count
    return jax.tree.unflatten(treedef, leaves)


class NullspacePolicy(StrictModule):
    """Known nullspaces plus explicit compatibility and gauge behavior."""

    right: LinearSubspace | None
    left: LinearSubspace | None
    certificate: KernelCertificate | None
    compatibility: CompatibilityMode = eqx.field(static=True)
    gauge: GaugeMode = eqx.field(static=True)

    def __init__(
        self,
        *,
        right: LinearSubspace | None = None,
        left: LinearSubspace | None = None,
        certificate: KernelCertificate | None = None,
        compatibility: CompatibilityMode = "error",
        gauge: GaugeMode = "minimum-norm",
    ):
        if certificate is not None and not isinstance(certificate, KernelCertificate):
            raise TypeError("certificate must be a KernelCertificate or None.")
        if certificate is not None:
            if right is None:
                right = certificate.right
            elif right.subspace_id != certificate.right.subspace_id:
                raise ValueError("Right nullspace does not match its kernel certificate.")
            if left is None:
                left = certificate.left
            elif (
                certificate.left is None
                or left.subspace_id != certificate.left.subspace_id
            ):
                raise ValueError("Left nullspace does not match its kernel certificate.")
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
        self.certificate = certificate
        self.compatibility = compatibility
        self.gauge = gauge


__all__ = [
    "CompatibilityMode",
    "GaugeMode",
    "LinearSubspace",
    "KernelCertificate",
    "NullspacePolicy",
]
