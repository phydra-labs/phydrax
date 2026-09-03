#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._high_resolution import HighResolutionReconstructionPlan


MHDReconstructionMethod: TypeAlias = Literal[
    "piecewise_constant",
    "plm",
    "weno_z",
    "teno",
    "mp5",
]


def _minmod(first: Array, second: Array, third: Array, /) -> Array:
    values = jnp.stack((first, second, third), axis=0)
    positive = jnp.all(values > 0.0, axis=0)
    negative = jnp.all(values < 0.0, axis=0)
    magnitude = jnp.min(jnp.abs(values), axis=0)
    return jnp.where(positive, magnitude, jnp.where(negative, -magnitude, 0.0))


class MHDPrimitiveReconstructionPlan(StrictModule, NonTrainableState):
    """Periodic primitive-variable MHD reconstruction with authoritative normal B."""

    method: MHDReconstructionMethod = eqx.field(static=True)
    plm_theta: float = eqx.field(static=True)
    high_resolution: HighResolutionReconstructionPlan | None
    characteristic_eigensystem: Callable | None = eqx.field(static=True)
    characteristic_id: str | None = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: MHDReconstructionMethod = "piecewise_constant",
        /,
        *,
        plm_theta: float = 1.5,
        order: int = 5,
        characteristic_eigensystem: Callable | None = None,
        characteristic_id: str | None = None,
    ):
        if method not in ("piecewise_constant", "plm", "weno_z", "teno", "mp5"):
            raise ValueError("Unknown MHD reconstruction method.")
        theta = float(plm_theta)
        if not np.isfinite(theta) or not 1.0 <= theta <= 2.0:
            raise ValueError("PLM theta must be finite and between one and two.")
        high_resolution = (
            HighResolutionReconstructionPlan(method, order=order)
            if method in ("weno_z", "teno", "mp5")
            else None
        )
        if (characteristic_eigensystem is None) != (characteristic_id is None):
            raise ValueError(
                "Characteristic eigensystem and identity must be supplied together."
            )
        if characteristic_eigensystem is not None and not callable(
            characteristic_eigensystem
        ):
            raise TypeError("Characteristic eigensystem must be callable.")
        self.method = method
        self.plm_theta = theta
        self.high_resolution = high_resolution
        self.characteristic_eigensystem = characteristic_eigensystem
        self.characteristic_id = characteristic_id
        self.reconstruction_id = canonical_fingerprint(
            {
                "kind": "mhd-primitive-reconstruction",
                "method": method,
                "plm_theta": theta,
                "high_resolution": (
                    None if high_resolution is None else high_resolution.plan_id
                ),
                "characteristic_id": characteristic_id,
            }
        )

    def _periodic_high_resolution(self, values: Array, /) -> tuple[Array, Array]:
        if self.high_resolution is None:
            raise RuntimeError("High-resolution reconstruction was not prepared.")
        radius = self.high_resolution.radius
        padded = jnp.concatenate((values[-radius:], values, values[:radius]), axis=0)
        left, right = self.high_resolution.reconstruct(padded)
        return left[radius:-radius], right[radius:-radius]

    def reconstruct(
        self,
        system: Any,
        full_state: Array,
        normal_field: Array,
        axis: int,
        *,
        lower_exterior: Array | None = None,
        upper_exterior: Array | None = None,
    ) -> tuple[Array, Array]:
        if tuple(system.component_names) != (
            "density",
            "momentum_x",
            "momentum_y",
            "momentum_z",
            "total_energy",
            "magnetic_x",
            "magnetic_y",
            "magnetic_z",
        ):
            raise TypeError("MHD primitive reconstruction requires canonical ideal MHD.")
        axis_ = int(axis)
        if axis_ < 0 or axis_ >= system.dimension:
            raise ValueError("MHD reconstruction axis is invalid.")
        primitive = system.conserved_to_primitive(full_state)
        values = jnp.moveaxis(primitive, axis_, 0)
        bounded = lower_exterior is not None or upper_exterior is not None
        if bounded:
            if lower_exterior is None or upper_exterior is None:
                raise ValueError("Both bounded MHD exterior states are required.")
            lower = system.conserved_to_primitive(lower_exterior)
            upper = system.conserved_to_primitive(upper_exterior)
            if lower.ndim == values.ndim - 1:
                lower = lower[None, ...]
                upper = upper[None, ...]
            extended = jnp.concatenate((lower, values, upper), axis=0)
            if self.method == "piecewise_constant":
                left = extended[:-1]
                right = extended[1:]
            elif self.method == "plm":
                backward = extended - jnp.roll(extended, 1, axis=0)
                forward = jnp.roll(extended, -1, axis=0) - extended
                centered = 0.5 * (
                    jnp.roll(extended, -1, axis=0) - jnp.roll(extended, 1, axis=0)
                )
                slope = _minmod(
                    self.plm_theta * backward,
                    centered,
                    self.plm_theta * forward,
                )
                slope = slope.at[0].set(0.0)
                slope = slope.at[-1].set(0.0)
                left = extended[:-1] + 0.5 * slope[:-1]
                right = extended[1:] - 0.5 * slope[1:]
            else:
                if self.high_resolution is None:
                    raise RuntimeError("High-resolution reconstruction was not prepared.")
                radius = self.high_resolution.radius
                padded = jnp.concatenate(
                    (
                        jnp.repeat(lower, radius, axis=0),
                        values,
                        jnp.repeat(upper, radius, axis=0),
                    ),
                    axis=0,
                )
                reconstructed_left, reconstructed_right = (
                    self.high_resolution.reconstruct(padded)
                )
                left = reconstructed_left[radius - 1 : radius + values.shape[0]]
                right = reconstructed_right[radius - 1 : radius + values.shape[0]]
        elif self.method == "piecewise_constant":
            left = values
            right = jnp.roll(values, -1, axis=0)
        elif self.method == "plm":
            backward = values - jnp.roll(values, 1, axis=0)
            forward = jnp.roll(values, -1, axis=0) - values
            centered = 0.5 * (jnp.roll(values, -1, axis=0) - jnp.roll(values, 1, axis=0))
            slope = _minmod(
                self.plm_theta * backward,
                centered,
                self.plm_theta * forward,
            )
            left = values + 0.5 * slope
            right = jnp.roll(values - 0.5 * slope, -1, axis=0)
        else:
            left, right = self._periodic_high_resolution(values)
        left = jnp.moveaxis(left, 0, axis_)
        right = jnp.moveaxis(right, 0, axis_)
        component = 5 + axis_
        left = left.at[..., component].set(normal_field)
        right = right.at[..., component].set(normal_field)
        left_conserved = system.primitive_to_conserved(left)
        right_conserved = system.primitive_to_conserved(right)
        if self.characteristic_eigensystem is None:
            return left_conserved, right_conserved
        left_matrix, right_matrix, _ = self.characteristic_eigensystem(
            left_conserved,
            right_conserved,
            axis_,
        )
        left_characteristic = ein.contract(
            "...ij,...j->...i", left_matrix, left_conserved
        )
        right_characteristic = ein.contract(
            "...ij,...j->...i", left_matrix, right_conserved
        )
        return (
            ein.contract("...ij,...j->...i", right_matrix, left_characteristic),
            ein.contract("...ij,...j->...i", right_matrix, right_characteristic),
        )


__all__ = ["MHDPrimitiveReconstructionPlan", "MHDReconstructionMethod"]
