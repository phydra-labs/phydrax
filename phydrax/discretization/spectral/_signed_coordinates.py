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

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._trainable import NonTrainableState
from ...linalg import AbstractRealCoordinateMap, ArraySpace, RealCoordinateEvidence


class SignedHermitianSpectralCoordinates(AbstractRealCoordinateMap, NonTrainableState):
    """Independent real coordinates for a masked signed-Hermitian mode layout.

    The represented subspace is ``c[i] = sign[i] * conj(c[partner[i]])`` on
    valid modes and zero on invalid padded storage. Signs are exactly ``+1`` or
    ``-1``; a fixed ``-1`` mode is therefore represented by one imaginary real
    coordinate. Component axes are carried independently and are never mixed.
    """

    conjugate_indices: Array
    conjugate_signs: Array
    valid_mask: Array
    fixed_indices: Array
    fixed_signs: Array
    representative_indices: Array
    partner_indices: Array
    coordinate_space: ArraySpace
    mode_shape: tuple[int, ...] = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    coordinate_size: int = eqx.field(static=True)
    maximum_coordinate_size: int = eqx.field(static=True)
    reality_tolerance: float = eqx.field(static=True)
    full_state_bytes: int = eqx.field(static=True)
    coordinate_state_bytes: int = eqx.field(static=True)
    fixed_mode_count: int = eqx.field(static=True)
    conjugate_pair_count: int = eqx.field(static=True)
    coordinate_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_shape: Sequence[int],
        conjugate_indices: ArrayLike,
        conjugate_signs: ArrayLike,
        /,
        *,
        valid_mask: ArrayLike | None = None,
        component_shape: Sequence[int] = (),
        coefficient_dtype: object = complex,
        layout_id: str,
        reality_tolerance: float = 1e-10,
        maximum_coordinate_size: int = 10_000_000,
    ):
        if any(isinstance(size, bool) for size in (*mode_shape, *component_shape)):
            raise TypeError("Mode and component dimensions must be integers.")
        modes = tuple(index(size) for size in mode_shape)
        components = tuple(index(size) for size in component_shape)
        if not modes or any(size <= 0 for size in modes + components):
            raise ValueError("Mode and component dimensions must be positive.")
        identifier_ = str(layout_id)
        if not identifier_:
            raise ValueError("layout_id must be non-empty.")
        if isinstance(maximum_coordinate_size, bool):
            raise TypeError("maximum_coordinate_size must be an integer.")
        maximum = index(maximum_coordinate_size)
        if maximum < 1:
            raise ValueError("maximum_coordinate_size must be positive.")
        tolerance = float(reality_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("reality_tolerance must be finite and nonnegative.")

        mode_size = prod(modes)
        partners_host = np.asarray(conjugate_indices, dtype=np.int64).reshape((-1,))
        signs_host = np.asarray(conjugate_signs).reshape((-1,))
        valid_host = (
            np.ones((mode_size,), dtype=bool)
            if valid_mask is None
            else np.asarray(valid_mask, dtype=bool).reshape((-1,))
        )
        if (
            partners_host.shape != (mode_size,)
            or signs_host.shape != (mode_size,)
            or valid_host.shape != (mode_size,)
        ):
            raise ValueError("Involution arrays must contain one entry per mode.")
        mode_indices = np.arange(mode_size, dtype=np.int64)
        if np.any(partners_host < 0) or np.any(partners_host >= mode_size):
            raise ValueError("conjugate_indices contains an out-of-range mode.")
        if not np.array_equal(partners_host[partners_host], mode_indices):
            raise ValueError("conjugate_indices must be an involution.")
        if not np.array_equal(valid_host[partners_host], valid_host):
            raise ValueError("valid_mask must be invariant under conjugation.")
        if np.iscomplexobj(signs_host) or not np.all(np.isin(signs_host, (-1, 1))):
            raise ValueError("conjugate_signs must contain only real +1 or -1 values.")
        signs_host = signs_host.astype(np.int8)
        if not np.array_equal(signs_host * signs_host[partners_host], np.ones(mode_size)):
            raise ValueError("Partner signs must compose to the identity.")

        component_count = prod(components) if components else 1
        state_shape = modes + components
        flat_indices = np.arange(mode_size * component_count, dtype=np.int64)
        modal_indices = flat_indices // component_count
        component_indices = flat_indices % component_count
        partners = partners_host[modal_indices] * component_count + component_indices
        signs = signs_host[modal_indices]
        valid = valid_host[modal_indices]
        fixed = flat_indices[valid & (partners == flat_indices)]
        representatives = flat_indices[valid & (flat_indices < partners)]
        pair_partners = partners[representatives]
        coordinate_size = int(fixed.size + 2 * representatives.size)
        if coordinate_size > maximum:
            raise ValueError(
                "Signed-Hermitian coordinates exceed maximum_coordinate_size."
            )

        dtype = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(coefficient_dtype)))
        if not jnp.issubdtype(dtype, jnp.complexfloating):
            raise TypeError(
                "Signed-Hermitian source coefficients must use a complex dtype."
            )
        coordinate_dtype = jnp.empty((), dtype=dtype).real.dtype
        identifier = canonical_fingerprint(
            {
                "kind": "signed-hermitian-spectral-coordinates-v1",
                "layout": identifier_,
                "mode_shape": list(modes),
                "component_shape": list(components),
                "partners": array_tree_fingerprint(partners_host),
                "signs": array_tree_fingerprint(signs_host),
                "valid": array_tree_fingerprint(valid_host),
                "dtype": dtype.str,
                "tolerance": tolerance,
            }
        )
        source_space = ArraySpace(
            state_shape,
            dtype=dtype,
            space_id=f"signed-hermitian-source:{identifier}",
        )
        coordinate_space = ArraySpace(
            (coordinate_size,),
            dtype=coordinate_dtype,
            space_id=f"signed-hermitian-coordinates:{identifier}",
        )
        evidence = RealCoordinateEvidence(
            domain_kind="constrained_subspace",
            source_space_id=source_space.space_id,
            coordinate_space_id=coordinate_space.space_id,
            source_dtype=str(dtype),
            coordinate_dtype=str(coordinate_dtype),
            source_shape=state_shape,
            coordinate_shape=(coordinate_size,),
            norm_relation="isometry",
            projection_kind="masked-signed-hermitian-orthogonal-v1",
            map_id=identifier,
        )
        self.source_space = source_space
        self.coordinate_space = coordinate_space
        self.evidence = evidence
        self.conjugate_indices = jnp.asarray(partners, dtype=jnp.int32)
        self.conjugate_signs = jnp.asarray(signs, dtype=jnp.int8)
        self.valid_mask = jnp.asarray(valid, dtype=bool)
        self.fixed_indices = jnp.asarray(fixed, dtype=jnp.int32)
        self.fixed_signs = jnp.asarray(signs[fixed], dtype=jnp.int8)
        self.representative_indices = jnp.asarray(representatives, dtype=jnp.int32)
        self.partner_indices = jnp.asarray(pair_partners, dtype=jnp.int32)
        self.mode_shape = modes
        self.component_shape = components
        self.state_shape = state_shape
        self.coordinate_size = coordinate_size
        self.maximum_coordinate_size = maximum
        self.reality_tolerance = tolerance
        self.full_state_bytes = mode_size * component_count * dtype.itemsize
        self.coordinate_state_bytes = coordinate_size * coordinate_dtype.itemsize
        self.fixed_mode_count = int(fixed.size)
        self.conjugate_pair_count = int(representatives.size)
        self.coordinate_id = identifier

    def validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError(
                f"Spectral state must have shape {self.state_shape}; got {value.shape}."
            )
        if not jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("Signed-Hermitian spectral state must be complex-valued.")
        return value

    def validate_coordinates(self, coordinates: ArrayLike, /) -> Array:
        value = self.coordinate_space.validate(coordinates)
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("Signed-Hermitian coordinates must be real-valued.")
        return value

    def defect(self, state: ArrayLike, /) -> Array:
        value = self.validate_state(state).reshape((-1,))
        signs = self.conjugate_signs.astype(value.dtype)
        expected = signs * jnp.conj(value[self.conjugate_indices])
        valid_defect = jnp.where(self.valid_mask, jnp.abs(value - expected), 0.0)
        padding_defect = jnp.where(self.valid_mask, 0.0, jnp.abs(value))
        return jnp.maximum(
            jnp.max(valid_defect, initial=0.0),
            jnp.max(padding_defect, initial=0.0),
        )

    def project(self, state: ArrayLike, /) -> Array:
        value = self.validate_state(state)
        flat = value.reshape((-1,))
        signs = self.conjugate_signs.astype(flat.dtype)
        projected = 0.5 * (flat + signs * jnp.conj(flat[self.conjugate_indices]))
        projected = jnp.where(self.valid_mask, projected, jnp.zeros((), flat.dtype))
        return projected.reshape(self.state_shape)

    def to_real_coordinates(self, state: ArrayLike, /) -> Array:
        value = self.validate_state(state)
        value = eqx.error_if(
            value,
            self.defect(value) > self.reality_tolerance,
            "Spectral state violates the masked signed-Hermitian reality contract.",
        )
        flat = value.reshape((-1,))
        fixed = flat[self.fixed_indices]
        fixed_values = jnp.where(
            self.fixed_signs > 0,
            jnp.real(fixed),
            jnp.imag(fixed),
        )
        pairs = flat[self.representative_indices]
        scale = jnp.sqrt(jnp.asarray(2.0, dtype=flat.real.dtype))
        return jnp.concatenate(
            (fixed_values, scale * jnp.real(pairs), scale * jnp.imag(pairs))
        )

    def from_real_coordinates(self, coordinates: ArrayLike, /) -> Array:
        values = self.validate_coordinates(coordinates)
        fixed_count = self.fixed_mode_count
        pair_count = self.conjugate_pair_count
        fixed_values = values[:fixed_count]
        real = values[fixed_count : fixed_count + pair_count]
        imaginary = values[fixed_count + pair_count :]
        scale = jnp.sqrt(jnp.asarray(2.0, dtype=values.dtype))
        pairs = jax.lax.complex(real / scale, imaginary / scale)
        dtype = self.source_space.dtype
        fixed = jax.lax.complex(
            jnp.where(self.fixed_signs > 0, fixed_values, 0.0),
            jnp.where(self.fixed_signs < 0, fixed_values, 0.0),
        )
        flat = jnp.zeros((prod(self.state_shape),), dtype=dtype)
        flat = flat.at[self.fixed_indices].set(fixed.astype(dtype))
        flat = flat.at[self.representative_indices].set(pairs.astype(dtype))
        partner_signs = self.conjugate_signs[self.partner_indices].astype(dtype)
        flat = flat.at[self.partner_indices].set(
            partner_signs * jnp.conj(pairs).astype(dtype)
        )
        return flat.reshape(self.state_shape)


__all__ = ["SignedHermitianSpectralCoordinates"]
