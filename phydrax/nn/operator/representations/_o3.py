#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import sqrt
from typing import Literal

import jax.numpy as jnp
from jax import Array, core as jax_core

import phydrax.ein as ein
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState


O3Parity = Literal[-1, 1]


def _tensor_basis(dtype: jnp.dtype, /) -> Array:
    inverse_root_two = 1.0 / sqrt(2.0)
    inverse_root_six = 1.0 / sqrt(6.0)
    return jnp.asarray(
        (
            (
                (inverse_root_two, 0.0, 0.0),
                (0.0, -inverse_root_two, 0.0),
                (0.0, 0.0, 0.0),
            ),
            (
                (inverse_root_six, 0.0, 0.0),
                (0.0, inverse_root_six, 0.0),
                (0.0, 0.0, -2.0 * inverse_root_six),
            ),
            ((0.0, inverse_root_two, 0.0), (inverse_root_two, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.0, 0.0, inverse_root_two), (0.0, 0.0, 0.0), (inverse_root_two, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, 0.0, inverse_root_two), (0.0, inverse_root_two, 0.0)),
        ),
        dtype=dtype,
    )


class O3Features(StrictModule):
    """Cartesian realization of scalar, vector, and rank-two O(3) irreps."""

    scalars: Array
    pseudoscalars: Array
    vectors: Array
    pseudovectors: Array
    tensors: Array
    pseudotensors: Array


class O3Representation(StrictModule, NonTrainableState):
    """Compact multiplicity schema for real O(3) irreducible field types."""

    scalars: int
    pseudoscalars: int
    vectors: int
    pseudovectors: int
    tensors: int
    pseudotensors: int

    def __init__(
        self,
        *,
        scalars: int = 0,
        pseudoscalars: int = 0,
        vectors: int = 0,
        pseudovectors: int = 0,
        tensors: int = 0,
        pseudotensors: int = 0,
    ) -> None:
        counts = tuple(
            (
                scalars,
                pseudoscalars,
                vectors,
                pseudovectors,
                tensors,
                pseudotensors,
            )
        )
        if any(value < 0 for value in counts) or not any(counts):
            raise ValueError(
                "O3Representation needs non-negative, non-empty multiplicities."
            )
        (
            self.scalars,
            self.pseudoscalars,
            self.vectors,
            self.pseudovectors,
            self.tensors,
            self.pseudotensors,
        ) = counts

    @property
    def packed_size(self) -> int:
        return (
            self.scalars
            + self.pseudoscalars
            + 3 * (self.vectors + self.pseudovectors)
            + 5 * (self.tensors + self.pseudotensors)
        )

    @property
    def channel_count(self) -> int:
        """Number of irreducible channels, the total multiplicity."""
        return (
            self.scalars
            + self.pseudoscalars
            + self.vectors
            + self.pseudovectors
            + self.tensors
            + self.pseudotensors
        )

    def _channel_blocks(self) -> tuple[tuple[int, int], ...]:
        # Packed order and orthonormal component widths of each irrep block.
        return (
            (self.scalars, 1),
            (self.pseudoscalars, 1),
            (self.vectors, 3),
            (self.pseudovectors, 3),
            (self.tensors, 5),
            (self.pseudotensors, 5),
        )

    def channel_squared_norms(self, values: Array, /) -> Array:
        """Squared Euclidean norm of every irreducible channel.

        Packed components are coordinates in orthonormal bases, so each value
        equals the Cartesian (Frobenius) squared norm of its channel and is
        invariant under every proper and improper orthogonal transform.
        """
        array = jnp.asarray(values)
        if array.shape[-1] != self.packed_size:
            raise ValueError(
                f"O(3) values require packed size {self.packed_size}; got {array.shape[-1]}."
            )
        offset = 0
        norms: list[Array] = []
        for count, width in self._channel_blocks():
            block = array[..., offset : offset + count * width]
            offset += count * width
            norms.append(
                jnp.sum(
                    jnp.square(block.reshape(array.shape[:-1] + (count, width))), axis=-1
                )
            )
        return jnp.concatenate(norms, axis=-1)

    def scale_channels(self, values: Array, factors: Array, /) -> Array:
        """Multiply every irreducible channel by one factor.

        The map is O(3)-equivariant whenever the factors are invariant.
        """
        array = jnp.asarray(values)
        scale = jnp.asarray(factors)
        if (
            array.shape[-1] != self.packed_size
            or scale.shape[-1] != self.channel_count
            or scale.shape[:-1] != array.shape[:-1]
        ):
            raise ValueError(
                "Channel scaling requires packed values and one factor per irreducible channel."
            )
        offset = 0
        channel = 0
        scaled: list[Array] = []
        for count, width in self._channel_blocks():
            block = array[..., offset : offset + count * width].reshape(
                array.shape[:-1] + (count, width)
            )
            factor = scale[..., channel : channel + count, None]
            scaled.append((block * factor).reshape(array.shape[:-1] + (count * width,)))
            offset += count * width
            channel += count
        return jnp.concatenate(scaled, axis=-1)

    def split(self, values: Array, /) -> O3Features:
        array = jnp.asarray(values)
        if array.shape[-1] != self.packed_size:
            raise ValueError(
                f"O(3) values require packed size {self.packed_size}; got {array.shape[-1]}."
            )
        offset = 0

        def take(count: int, width: int = 1) -> Array:
            nonlocal offset
            size = count * width
            selected = array[..., offset : offset + size]
            offset += size
            if width == 1:
                return selected
            return selected.reshape(array.shape[:-1] + (count, width))

        scalars = take(self.scalars)
        pseudoscalars = take(self.pseudoscalars)
        vectors = take(self.vectors, 3)
        pseudovectors = take(self.pseudovectors, 3)
        tensor_coefficients = take(self.tensors, 5)
        pseudotensor_coefficients = take(self.pseudotensors, 5)
        basis = _tensor_basis(array.dtype)
        tensors = ein.contract("...mk,kij->...mij", tensor_coefficients, basis)
        pseudotensors = ein.contract(
            "...mk,kij->...mij", pseudotensor_coefficients, basis
        )
        return O3Features(
            scalars=scalars,
            pseudoscalars=pseudoscalars,
            vectors=vectors,
            pseudovectors=pseudovectors,
            tensors=tensors,
            pseudotensors=pseudotensors,
        )

    def join(self, features: O3Features, /) -> Array:
        basis = _tensor_basis(features.scalars.dtype)
        tensor_coefficients = ein.contract("...mij,kij->...mk", features.tensors, basis)
        pseudotensor_coefficients = ein.contract(
            "...mij,kij->...mk", features.pseudotensors, basis
        )
        parts = (
            features.scalars,
            features.pseudoscalars,
            features.vectors.reshape(features.vectors.shape[:-2] + (3 * self.vectors,)),
            features.pseudovectors.reshape(
                features.pseudovectors.shape[:-2] + (3 * self.pseudovectors,)
            ),
            tensor_coefficients.reshape(
                tensor_coefficients.shape[:-2] + (5 * self.tensors,)
            ),
            pseudotensor_coefficients.reshape(
                pseudotensor_coefficients.shape[:-2] + (5 * self.pseudotensors,)
            ),
        )
        return jnp.concatenate(parts, axis=-1)

    def transform(self, values: Array, orthogonal: Array, /) -> Array:
        """Apply an orthogonal 3-D frame transform to packed field values."""
        matrix = jnp.asarray(orthogonal)
        if matrix.shape != (3, 3):
            raise ValueError("O(3) transforms require a (3, 3) matrix.")
        if not isinstance(matrix, jax_core.Tracer):
            error = jnp.max(jnp.abs(matrix.T @ matrix - jnp.eye(3, dtype=matrix.dtype)))
            if float(error) > 1e-6:
                raise ValueError("O(3) transform matrix must be orthogonal.")
        determinant = jnp.linalg.det(matrix)
        features = self.split(values)
        vectors = ein.contract("ij,...mj->...mi", matrix, features.vectors)
        pseudovectors = determinant * ein.contract(
            "ij,...mj->...mi", matrix, features.pseudovectors
        )
        tensors = ein.contract("ia,...mab,jb->...mij", matrix, features.tensors, matrix)
        pseudotensors = determinant * ein.contract(
            "ia,...mab,jb->...mij", matrix, features.pseudotensors, matrix
        )
        return self.join(
            O3Features(
                scalars=features.scalars,
                pseudoscalars=determinant * features.pseudoscalars,
                vectors=vectors,
                pseudovectors=pseudovectors,
                tensors=tensors,
                pseudotensors=pseudotensors,
            )
        )
