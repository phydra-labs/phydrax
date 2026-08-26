#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from operator import index

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace
from ._space import TensorSpectralDiscretization


class HermitianSpectralCoordinates(StrictModule, NonTrainableState):
    """Independent real coordinates for a real field in full complex storage."""

    discretization: TensorSpectralDiscretization
    conjugate_indices: Array
    fixed_indices: Array
    representative_indices: Array
    partner_indices: Array
    coordinate_space: ArraySpace
    state_shape: tuple[int, ...] = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    coordinate_size: int = eqx.field(static=True)
    maximum_coordinate_size: int = eqx.field(static=True)
    reality_tolerance: float = eqx.field(static=True)
    coordinate_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        component_shape: Sequence[int] = (),
        reality_tolerance: float = 1e-10,
        maximum_coordinate_size: int = 10_000_000,
    ):
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        if any(isinstance(size, bool) for size in component_shape):
            raise TypeError("component_shape dimensions must be integers.")
        components = tuple(index(size) for size in component_shape)
        if any(size <= 0 for size in components):
            raise ValueError("component_shape dimensions must be positive.")
        if isinstance(maximum_coordinate_size, bool):
            raise TypeError("maximum_coordinate_size must be an integer.")
        maximum = index(maximum_coordinate_size)
        if maximum < 1:
            raise ValueError("maximum_coordinate_size must be positive.")
        tolerance = float(reality_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("reality_tolerance must be finite and nonnegative.")
        modal_shape = discretization.modal_shape
        multi_indices = np.indices(modal_shape, dtype=np.int64).reshape(
            (len(modal_shape), -1)
        )
        conjugate_multi = np.stack(
            tuple(
                np.asarray(axis.modes.conjugate_indices, dtype=np.int64)[
                    multi_indices[axis_index]
                ]
                for axis_index, axis in enumerate(discretization.axes)
            ),
            axis=0,
        )
        modal_conjugates = np.ravel_multi_index(conjugate_multi, modal_shape)
        component_count = prod(components) if components else 1
        modal_size = prod(modal_shape)
        if modal_size * component_count > maximum:
            raise ValueError(
                "Hermitian coordinate count exceeds maximum_coordinate_size."
            )
        flat_indices = np.arange(modal_size * component_count, dtype=np.int64)
        modal_indices = flat_indices // component_count
        component_indices = flat_indices % component_count
        conjugates = modal_conjugates[modal_indices] * component_count + component_indices
        fixed = flat_indices[conjugates == flat_indices]
        representatives = flat_indices[flat_indices < conjugates]
        partners = conjugates[representatives]
        coordinate_size = int(fixed.size + 2 * representatives.size)
        coefficient_dtype = jnp.dtype(discretization.plan.precision.coefficient_dtype)
        coordinate_dtype = jnp.empty((), dtype=coefficient_dtype).real.dtype
        state_shape = modal_shape + components
        identifier = canonical_fingerprint(
            {
                "kind": "hermitian-spectral-coordinates-v1",
                "discretization": discretization.prepared_id,
                "component_shape": list(components),
                "state_shape": list(state_shape),
                "coordinate_size": coordinate_size,
                "reality_tolerance": tolerance,
                "maximum_coordinate_size": maximum,
            }
        )
        self.discretization = discretization
        self.conjugate_indices = jnp.asarray(conjugates, dtype=jnp.int32)
        self.fixed_indices = jnp.asarray(fixed, dtype=jnp.int32)
        self.representative_indices = jnp.asarray(representatives, dtype=jnp.int32)
        self.partner_indices = jnp.asarray(partners, dtype=jnp.int32)
        self.coordinate_space = ArraySpace(
            (coordinate_size,),
            dtype=coordinate_dtype,
            space_id=f"hermitian-coordinates:{identifier}",
        )
        self.state_shape = state_shape
        self.component_shape = components
        self.coordinate_size = coordinate_size
        self.maximum_coordinate_size = maximum
        self.reality_tolerance = tolerance
        self.coordinate_id = identifier

    def validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError(
                f"Spectral state must have shape {self.state_shape}; got {value.shape}."
            )
        if not jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("Hermitian spectral state must be complex-valued.")
        return value

    def validate_coordinates(self, coordinates: ArrayLike, /) -> Array:
        value = self.coordinate_space.validate(coordinates)
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("Hermitian analysis coordinates must be real-valued.")
        return value

    def reality_defect(self, state: ArrayLike, /) -> Array:
        value = self.validate_state(state).reshape((-1,))
        expected = jnp.conj(value[self.conjugate_indices])
        return jnp.max(jnp.abs(value - expected), initial=0.0)

    def project(self, state: ArrayLike, /) -> Array:
        """Orthogonally project onto the Hermitian real-field subspace."""
        value = self.validate_state(state)
        flat = value.reshape((-1,))
        projected = 0.5 * flat + 0.5 * jnp.conj(flat[self.conjugate_indices])
        return projected.reshape(self.state_shape)

    def to_real_coordinates(self, state: ArrayLike, /) -> Array:
        value = self.validate_state(state)
        defect = self.reality_defect(value)
        value = eqx.error_if(
            value,
            defect > self.reality_tolerance,
            "Spectral state violates the prepared Hermitian reality contract.",
        )
        flat = value.reshape((-1,))
        scale = jnp.sqrt(jnp.asarray(2.0, dtype=flat.real.dtype))
        return jnp.concatenate(
            (
                jnp.real(flat[self.fixed_indices]),
                scale * jnp.real(flat[self.representative_indices]),
                scale * jnp.imag(flat[self.representative_indices]),
            )
        )

    def from_real_coordinates(self, coordinates: ArrayLike, /) -> Array:
        values = self.validate_coordinates(coordinates)
        fixed_count = int(self.fixed_indices.size)
        pair_count = int(self.representative_indices.size)
        fixed = values[:fixed_count]
        real = values[fixed_count : fixed_count + pair_count]
        imaginary = values[fixed_count + pair_count :]
        scale = jnp.sqrt(jnp.asarray(2.0, dtype=values.dtype))
        pairs = jax.lax.complex(real / scale, imaginary / scale)
        dtype = jnp.dtype(self.discretization.plan.precision.coefficient_dtype)
        flat = jnp.zeros((prod(self.state_shape),), dtype=dtype)
        flat = flat.at[self.fixed_indices].set(fixed.astype(dtype))
        flat = flat.at[self.representative_indices].set(pairs.astype(dtype))
        flat = flat.at[self.partner_indices].set(jnp.conj(pairs).astype(dtype))
        return flat.reshape(self.state_shape)


__all__ = ["HermitianSpectralCoordinates"]
