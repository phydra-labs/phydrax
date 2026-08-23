#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


class MatrixProductState(StrictModule):
    tensors: tuple[Array, ...]
    site_count: int
    physical_dimensions: tuple[int, ...]
    bond_dimensions: tuple[int, ...]

    def __init__(self, tensors: Sequence[ArrayLike], /):
        values = tuple(jnp.asarray(tensor) for tensor in tensors)
        if not values or any(tensor.ndim != 3 for tensor in values):
            raise ValueError("MPS tensors must have shape (left, physical, right).")
        if values[0].shape[0] != 1 or values[-1].shape[-1] != 1:
            raise ValueError("Open-boundary MPS edge bonds must be one.")
        for left, right in zip(values[:-1], values[1:], strict=True):
            if left.shape[-1] != right.shape[0]:
                raise ValueError("Adjacent MPS bond dimensions must match.")
        self.tensors = values
        self.site_count = len(values)
        self.physical_dimensions = tuple(int(tensor.shape[1]) for tensor in values)
        self.bond_dimensions = tuple(int(tensor.shape[-1]) for tensor in values[:-1])

    def to_dense(self) -> Array:
        state = self.tensors[0][0]
        for tensor in self.tensors[1:]:
            state = jnp.tensordot(state, tensor, axes=(-1, 0))
        return state[..., 0].reshape(-1)

    def norm(self) -> Array:
        return jnp.linalg.norm(self.to_dense())

    def normalized(self) -> MatrixProductState:
        norm = self.norm()
        tensors = (self.tensors[0] / norm,) + self.tensors[1:]
        return MatrixProductState(tensors)


class MatrixProductOperator(StrictModule):
    tensors: tuple[Array, ...]
    site_count: int
    output_dimensions: tuple[int, ...]
    input_dimensions: tuple[int, ...]

    def __init__(self, tensors: Sequence[ArrayLike], /):
        values = tuple(jnp.asarray(tensor) for tensor in tensors)
        if not values or any(tensor.ndim != 4 for tensor in values):
            raise ValueError("MPO tensors require (left, output, input, right).")
        if values[0].shape[0] != 1 or values[-1].shape[-1] != 1:
            raise ValueError("Open-boundary MPO edge bonds must be one.")
        for left, right in zip(values[:-1], values[1:], strict=True):
            if left.shape[-1] != right.shape[0]:
                raise ValueError("Adjacent MPO bonds must match.")
        self.tensors = values
        self.site_count = len(values)
        self.output_dimensions = tuple(int(tensor.shape[1]) for tensor in values)
        self.input_dimensions = tuple(int(tensor.shape[2]) for tensor in values)

    def to_dense(self) -> Array:
        operator = self.tensors[0][0]
        for tensor in self.tensors[1:]:
            operator = jnp.tensordot(operator, tensor, axes=(-1, 0))
        operator = operator[..., 0]
        output_axes = tuple(range(0, 2 * self.site_count, 2))
        input_axes = tuple(range(1, 2 * self.site_count, 2))
        operator = jnp.transpose(operator, output_axes + input_axes)
        return operator.reshape((prod(self.output_dimensions), prod(self.input_dimensions)))


class LocallyPurifiedDensity(StrictModule):
    tensors: tuple[Array, ...]
    site_count: int
    physical_dimensions: tuple[int, ...]
    purification_dimensions: tuple[int, ...]

    def __init__(self, tensors: Sequence[ArrayLike], /):
        values = tuple(jnp.asarray(tensor) for tensor in tensors)
        if not values or any(tensor.ndim != 4 for tensor in values):
            raise ValueError("Purification tensors require (left, physical, kraus, right).")
        if values[0].shape[0] != 1 or values[-1].shape[-1] != 1:
            raise ValueError("Purification edge bonds must be one.")
        for left, right in zip(values[:-1], values[1:], strict=True):
            if left.shape[-1] != right.shape[0]:
                raise ValueError("Adjacent purification bonds must match.")
        self.tensors = values
        self.site_count = len(values)
        self.physical_dimensions = tuple(int(tensor.shape[1]) for tensor in values)
        self.purification_dimensions = tuple(int(tensor.shape[2]) for tensor in values)

    def amplitude(self) -> Array:
        amplitude = self.tensors[0][0]
        for tensor in self.tensors[1:]:
            amplitude = jnp.tensordot(amplitude, tensor, axes=(-1, 0))
        amplitude = amplitude[..., 0]
        physical_axes = tuple(range(0, 2 * self.site_count, 2))
        kraus_axes = tuple(range(1, 2 * self.site_count, 2))
        amplitude = jnp.transpose(amplitude, physical_axes + kraus_axes)
        return amplitude.reshape(
            (prod(self.physical_dimensions), prod(self.purification_dimensions))
        )

    def density(self) -> Array:
        amplitude = self.amplitude()
        density = amplitude @ jnp.conj(amplitude.T)
        return density / jnp.trace(density)


# Local import avoids exposing a utility dependency in public signatures.
from math import prod


__all__ = ["LocallyPurifiedDensity", "MatrixProductOperator", "MatrixProductState"]
