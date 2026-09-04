#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import AbstractVectorSpace, ArraySpace, DualSpace
from ._state_geometry import AbstractStateGeometry


def _shape(value: Sequence[int], owner: str, /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{owner} dimensions must be positive.")
    return shape


def _array_space_shape(space: AbstractVectorSpace, owner: str, /) -> tuple[int, ...]:
    if not isinstance(space, AbstractVectorSpace):
        raise TypeError(f"{owner} must be an AbstractVectorSpace.")
    structure = space.structure()
    if not isinstance(structure, jax.ShapeDtypeStruct):
        raise TypeError(f"{owner} must describe exactly one array.")
    return tuple(structure.shape)


class ProductStateGeometryBlock(StrictModule, NonTrainableState):
    """One point block with explicit local and physical tangent spaces."""

    geometry: AbstractStateGeometry
    local_space: AbstractVectorSpace
    tangent_space: AbstractVectorSpace
    local_cotangent_space: DualSpace
    cotangent_space: DualSpace
    point_shape: tuple[int, ...] = eqx.field(static=True)
    local_shape: tuple[int, ...] = eqx.field(static=True)
    tangent_shape: tuple[int, ...] = eqx.field(static=True)
    point_size: int = eqx.field(static=True)
    local_size: int = eqx.field(static=True)
    tangent_size: int = eqx.field(static=True)
    cotangent_size: int = eqx.field(static=True)
    block_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: AbstractStateGeometry,
        point_shape: Sequence[int],
        /,
        *,
        block_id: str,
        local_space: AbstractVectorSpace | None = None,
        tangent_space: AbstractVectorSpace | None = None,
    ):
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("geometry must be an AbstractStateGeometry.")
        point_shape_ = _shape(point_shape, "Product-state point block")
        local_space_ = ArraySpace(point_shape_) if local_space is None else local_space
        tangent_space_ = (
            ArraySpace(point_shape_) if tangent_space is None else tangent_space
        )
        local_shape_ = _array_space_shape(local_space_, "local_space")
        tangent_shape_ = _array_space_shape(tangent_space_, "tangent_space")
        identifier = str(block_id)
        if not identifier:
            raise ValueError("block_id must be nonempty.")
        self.geometry = geometry
        self.local_space = local_space_
        self.tangent_space = tangent_space_
        self.local_cotangent_space = DualSpace(local_space_)
        self.cotangent_space = DualSpace(tangent_space_)
        self.point_shape = point_shape_
        self.local_shape = local_shape_
        self.tangent_shape = tangent_shape_
        self.point_size = prod(point_shape_) if point_shape_ else 1
        self.local_size = local_space_.size
        self.tangent_size = tangent_space_.size
        self.cotangent_size = tangent_space_.size
        self.block_id = identifier


def _offsets(sizes: Sequence[int], /) -> tuple[int, ...]:
    values = [0]
    for size in sizes:
        values.append(values[-1] + size)
    return tuple(values)


class ProductStateGeometry(AbstractStateGeometry):
    """Flat storage for a product with independent four-space block layouts."""

    blocks: tuple[ProductStateGeometryBlock, ...]
    point_offsets: tuple[int, ...] = eqx.field(static=True)
    local_offsets: tuple[int, ...] = eqx.field(static=True)
    tangent_offsets: tuple[int, ...] = eqx.field(static=True)
    cotangent_offsets: tuple[int, ...] = eqx.field(static=True)
    point_size: int = eqx.field(static=True)
    local_size: int = eqx.field(static=True)
    tangent_size: int = eqx.field(static=True)
    cotangent_size: int = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(
        self,
        blocks: Sequence[ProductStateGeometryBlock],
        /,
        *,
        geometry_id: str | None = None,
    ):
        blocks_ = tuple(blocks)
        if not blocks_ or any(
            not isinstance(block, ProductStateGeometryBlock) for block in blocks_
        ):
            raise TypeError("blocks must contain ProductStateGeometryBlock values.")
        if len({block.block_id for block in blocks_}) != len(blocks_):
            raise ValueError("Product-state block IDs must be unique.")
        point_offsets = _offsets(tuple(block.point_size for block in blocks_))
        local_offsets = _offsets(tuple(block.local_size for block in blocks_))
        tangent_offsets = _offsets(tuple(block.tangent_size for block in blocks_))
        cotangent_offsets = _offsets(tuple(block.cotangent_size for block in blocks_))
        identifier = geometry_id or canonical_fingerprint(
            {
                "kind": "four-space-product-state-geometry",
                "blocks": [
                    {
                        "id": block.block_id,
                        "point_shape": list(block.point_shape),
                        "local_space": block.local_space.space_id,
                        "tangent_space": block.tangent_space.space_id,
                        "geometry": block.geometry.geometry_id,
                    }
                    for block in blocks_
                ],
            }
        )
        self.blocks = blocks_
        self.point_offsets = point_offsets
        self.local_offsets = local_offsets
        self.tangent_offsets = tangent_offsets
        self.cotangent_offsets = cotangent_offsets
        self.point_size = point_offsets[-1]
        self.local_size = local_offsets[-1]
        self.tangent_size = tangent_offsets[-1]
        self.cotangent_size = cotangent_offsets[-1]
        self.geometry_id = str(identifier)
        self.retraction_method = "four-space-product"
        self.trivial = all(block.geometry.trivial for block in blocks_)
        self.supports_exact_inverse = all(
            block.geometry.supports_exact_inverse for block in blocks_
        )
        self.supports_exact_differential = all(
            block.geometry.supports_exact_differential for block in blocks_
        )
        self.supports_transport = all(
            block.geometry.supports_transport for block in blocks_
        )
        self.supports_isometric_transport = all(
            block.geometry.supports_isometric_transport for block in blocks_
        )
        self.supports_commutator_free = all(
            block.geometry.supports_commutator_free for block in blocks_
        )

    @staticmethod
    def _flat(value: ArrayLike, size: int, name: str, /) -> Array:
        array = jnp.asarray(value)
        if array.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},); got {array.shape}.")
        return array

    @staticmethod
    def _split(
        value: ArrayLike,
        shapes: tuple[tuple[int, ...], ...],
        offsets: tuple[int, ...],
        size: int,
        name: str,
        /,
    ) -> tuple[Array, ...]:
        flat = ProductStateGeometry._flat(value, size, name)
        return tuple(
            flat[left:right].reshape(shape)
            for shape, left, right in zip(shapes, offsets[:-1], offsets[1:], strict=True)
        )

    @staticmethod
    def _combine(
        values: Sequence[ArrayLike],
        shapes: tuple[tuple[int, ...], ...],
        name: str,
        /,
    ) -> Array:
        values_ = tuple(values)
        if len(values_) != len(shapes):
            raise ValueError(f"{name} values must match the block count.")
        flattened = []
        for index, (value, shape) in enumerate(zip(values_, shapes, strict=True)):
            array = jnp.asarray(value)
            if array.shape != shape:
                raise ValueError(
                    f"{name} block {index} must have shape {shape}; got {array.shape}."
                )
            flattened.append(array.reshape((-1,)))
        return jnp.concatenate(tuple(flattened))

    @property
    def _point_shapes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(block.point_shape for block in self.blocks)

    @property
    def _local_shapes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(block.local_shape for block in self.blocks)

    @property
    def _tangent_shapes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(block.tangent_shape for block in self.blocks)

    def split_point(self, value: ArrayLike, /) -> tuple[Array, ...]:
        return self._split(
            value,
            self._point_shapes,
            self.point_offsets,
            self.point_size,
            "Product point",
        )

    def combine_point(self, values: Sequence[ArrayLike], /) -> Array:
        return self._combine(values, self._point_shapes, "Product point")

    def split_local(self, value: ArrayLike, /) -> tuple[Array, ...]:
        return self._split(
            value,
            self._local_shapes,
            self.local_offsets,
            self.local_size,
            "Product local tangent",
        )

    def combine_local(self, values: Sequence[ArrayLike], /) -> Array:
        return self._combine(values, self._local_shapes, "Product local tangent")

    def split_tangent(self, value: ArrayLike, /) -> tuple[Array, ...]:
        return self._split(
            value,
            self._tangent_shapes,
            self.tangent_offsets,
            self.tangent_size,
            "Product physical tangent",
        )

    def combine_tangent(self, values: Sequence[ArrayLike], /) -> Array:
        return self._combine(values, self._tangent_shapes, "Product physical tangent")

    def split_local_cotangent(self, value: ArrayLike, /) -> tuple[Array, ...]:
        return self._split(
            value,
            self._local_shapes,
            self.local_offsets,
            self.local_size,
            "Product local cotangent",
        )

    def combine_local_cotangent(self, values: Sequence[ArrayLike], /) -> Array:
        return self._combine(values, self._local_shapes, "Product local cotangent")

    def split_cotangent(self, value: ArrayLike, /) -> tuple[Array, ...]:
        return self._split(
            value,
            self._tangent_shapes,
            self.cotangent_offsets,
            self.cotangent_size,
            "Product physical cotangent",
        )

    def combine_cotangent(self, values: Sequence[ArrayLike], /) -> Array:
        return self._combine(values, self._tangent_shapes, "Product physical cotangent")

    def contains(self, state: ArrayLike, /) -> Array:
        valid = jnp.asarray(True)
        for block, value in zip(self.blocks, self.split_point(state), strict=True):
            valid = valid & block.geometry.contains(value)
        return valid

    def project_tangent(self, state: ArrayLike, vector: ArrayLike, /) -> Array:
        states = self.split_point(state)
        vectors = self.split_point(vector)
        return self.combine_tangent(
            tuple(
                block.geometry.project_tangent(point, ambient)
                for block, point, ambient in zip(
                    self.blocks, states, vectors, strict=True
                )
            )
        )

    def retract(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        states = self.split_point(state)
        locals_ = self.split_local(local_tangent)
        return self.combine_point(
            tuple(
                block.geometry.retract(point, local)
                for block, point, local in zip(self.blocks, states, locals_, strict=True)
            )
        )

    def inverse_retract(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        states = self.split_point(state)
        points = self.split_point(point)
        return self.combine_local(
            tuple(
                block.geometry.inverse_retract(left, right)
                for block, left, right in zip(self.blocks, states, points, strict=True)
            )
        )

    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        /,
    ) -> Array:
        states = self.split_point(state)
        locals_ = self.split_local(local_tangent)
        directions = self.split_local(local_velocity)
        return self.combine_tangent(
            tuple(
                block.geometry.retraction_jvp(point, local, direction)
                for block, point, local, direction in zip(
                    self.blocks, states, locals_, directions, strict=True
                )
            )
        )

    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        states = self.split_point(state)
        points = self.split_point(point)
        tangents = self.split_tangent(tangent)
        return self.combine_local(
            tuple(
                block.geometry.retraction_inverse_jvp(left, right, vector)
                for block, left, right, vector in zip(
                    self.blocks, states, points, tangents, strict=True
                )
            )
        )

    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        states = self.split_point(state)
        locals_ = self.split_local(local_tangent)
        cotangents = self.split_cotangent(cotangent)
        return self.combine_local_cotangent(
            tuple(
                block.geometry.retraction_vjp(point, local, covector)
                for block, point, local, covector in zip(
                    self.blocks, states, locals_, cotangents, strict=True
                )
            )
        )

    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        states = self.split_point(state)
        points = self.split_point(point)
        tangents = self.split_tangent(tangent)
        return self.combine_tangent(
            tuple(
                block.geometry.transport_tangent(left, right, vector)
                for block, left, right, vector in zip(
                    self.blocks, states, points, tangents, strict=True
                )
            )
        )

    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        states = self.split_point(state)
        points = self.split_point(point)
        cotangents = self.split_cotangent(cotangent)
        return self.combine_cotangent(
            tuple(
                block.geometry.transport_cotangent_pullback(left, right, covector)
                for block, left, right, covector in zip(
                    self.blocks, states, points, cotangents, strict=True
                )
            )
        )

    def cut_locus_margin(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        states = self.split_point(state)
        points = self.split_point(point)
        margins = tuple(
            block.geometry.cut_locus_margin(left, right)
            for block, left, right in zip(self.blocks, states, points, strict=True)
        )
        return jnp.min(jnp.stack(margins))


__all__ = ["ProductStateGeometry", "ProductStateGeometryBlock"]
