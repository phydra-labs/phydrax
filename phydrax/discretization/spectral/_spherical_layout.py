#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class SphericalModeLayout(StrictModule, NonTrainableState):
    """Spin-spherical ``(ell, m)`` modes and their padded S2FFT storage."""

    degrees: Array
    orders: Array
    valid_mask: Array
    independent_mask: Array
    conjugate_indices: Array
    conjugate_signs: Array
    valid_indices: Array
    bandlimit: int = eqx.field(static=True)
    spin: int = eqx.field(static=True)
    reality: bool = eqx.field(static=True)
    coefficient_shape: tuple[int, int] = eqx.field(static=True)
    logical_mode_count: int = eqx.field(static=True)
    level_multiplicities: tuple[int, ...] = eqx.field(static=True)
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    normalization: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        bandlimit: int,
        /,
        *,
        spin: int = 0,
        reality: bool = True,
    ):
        limit = int(bandlimit)
        spin_ = int(spin)
        reality_ = bool(reality)
        if limit <= abs(spin_):
            raise ValueError("bandlimit must exceed the absolute spin.")
        if reality_ and spin_ != 0:
            raise ValueError("Real spherical layouts require spin zero.")
        degree = np.arange(limit, dtype=np.int32)[:, None]
        order = np.arange(-(limit - 1), limit, dtype=np.int32)[None, :]
        degrees = np.broadcast_to(degree, (limit, 2 * limit - 1)).copy()
        orders = np.broadcast_to(order, (limit, 2 * limit - 1)).copy()
        valid = (np.abs(orders) <= degrees) & (degrees >= abs(spin_))
        independent = valid & (orders >= 0) if reality_ else valid.copy()
        conjugates = np.arange(2 * limit - 1, dtype=np.int32)[::-1].copy()
        signs = (-1.0) ** np.abs(np.arange(-(limit - 1), limit, dtype=np.int32))
        indices = np.flatnonzero(valid.reshape((-1,))).astype(np.int32)
        mode_ids = tuple(
            f"spin:{spin_}:ell:{int(ell)}:m:{int(m)}"
            for ell, m, active in zip(
                degrees.reshape((-1,)),
                orders.reshape((-1,)),
                valid.reshape((-1,)),
                strict=True,
            )
            if active
        )
        multiplicities = tuple(
            0 if ell < abs(spin_) else 2 * ell + 1 for ell in range(limit)
        )
        normalization = "s2fft-orthonormal-condon-shortley"
        self.degrees = jnp.asarray(degrees)
        self.orders = jnp.asarray(orders)
        self.valid_mask = jnp.asarray(valid)
        self.independent_mask = jnp.asarray(independent)
        self.conjugate_indices = jnp.asarray(conjugates)
        self.conjugate_signs = jnp.asarray(signs)
        self.valid_indices = jnp.asarray(indices)
        self.bandlimit = limit
        self.spin = spin_
        self.reality = reality_
        self.coefficient_shape = (limit, 2 * limit - 1)
        self.logical_mode_count = int(indices.size)
        self.level_multiplicities = multiplicities
        self.mode_ids = mode_ids
        self.normalization = normalization
        self.layout_id = canonical_fingerprint(
            {
                "kind": "spherical-mode-layout-v1",
                "bandlimit": limit,
                "spin": spin_,
                "reality": reality_,
                "valid": array_tree_fingerprint(valid),
                "independent": array_tree_fingerprint(independent),
                "normalization": normalization,
            }
        )

    def _coefficient_axes(self, array: Array, /) -> tuple[int, int, bool]:
        if array.ndim >= 2 and tuple(array.shape[-2:]) == self.coefficient_shape:
            return array.ndim - 2, array.ndim - 1, False
        if (
            array.ndim >= 3
            and tuple(array.shape[-3:-1]) == self.coefficient_shape
        ):
            return array.ndim - 3, array.ndim - 2, True
        raise ValueError(
            "Spherical coefficients must end in (ell, m) or (ell, m, channels) "
            f"with mode shape {self.coefficient_shape}; got {array.shape}."
        )

    def _broadcast_mode_array(
        self,
        value: Array,
        array: Array,
        /,
        *,
        channel_last: bool,
    ) -> Array:
        trailing = 1 if channel_last else 0
        leading = array.ndim - 2 - trailing
        return value.reshape((1,) * leading + self.coefficient_shape + (1,) * trailing)

    def mask_invalid(self, coefficients: ArrayLike, /) -> Array:
        """Replace invalid padded storage before it can participate in arithmetic."""
        array = jnp.asarray(coefficients)
        _, _, channel_last = self._coefficient_axes(array)
        mask = self._broadcast_mode_array(
            self.valid_mask,
            array,
            channel_last=channel_last,
        )
        return jnp.where(mask, array, jnp.zeros((), dtype=array.dtype))

    def canonicalize_reality(self, coefficients: ArrayLike, /) -> Array:
        """Fill negative orders from the independent real-field coefficients."""
        if not self.reality:
            raise ValueError("Reality canonicalization requires a real spherical layout.")
        array = self.mask_invalid(coefficients)
        _, order_axis, channel_last = self._coefficient_axes(array)
        mirrored = jnp.take(array, self.conjugate_indices, axis=order_axis)
        sign = self.conjugate_signs
        sign_shape = [1] * array.ndim
        sign_shape[order_axis] = sign.size
        mirrored = jnp.conj(mirrored) * sign.reshape(tuple(sign_shape)).astype(array.dtype)
        negative = self._broadcast_mode_array(
            self.valid_mask & (self.orders < 0),
            array,
            channel_last=channel_last,
        )
        return self.mask_invalid(jnp.where(negative, mirrored, array))

    def conjugacy_defect(self, coefficients: ArrayLike, /) -> Array:
        """Return the largest active negative-order real-field symmetry defect."""
        if not self.reality:
            raise ValueError("Conjugacy defects require a real spherical layout.")
        array = jnp.asarray(coefficients)
        canonical = self.canonicalize_reality(array)
        _, _, channel_last = self._coefficient_axes(array)
        negative = self._broadcast_mode_array(
            self.valid_mask & (self.orders < 0),
            array,
            channel_last=channel_last,
        )
        difference = jnp.where(
            negative,
            jnp.abs(array - canonical),
            jnp.zeros((), dtype=jnp.real(array).dtype),
        )
        return jnp.max(difference, initial=0.0)

    def level_values(self, values: ArrayLike, /) -> Array:
        """Broadcast one scalar per degree into valid padded coefficient storage."""
        levels = jnp.asarray(values)
        if levels.shape != (self.bandlimit,):
            raise ValueError(
                f"Spherical level values must have shape {(self.bandlimit,)}; "
                f"got {levels.shape}."
            )
        padded = jnp.broadcast_to(levels[:, None], self.coefficient_shape)
        return jnp.where(self.valid_mask, padded, jnp.zeros((), dtype=padded.dtype))


__all__ = ["SphericalModeLayout"]
