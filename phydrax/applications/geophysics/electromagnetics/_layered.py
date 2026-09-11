#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....special import jv


VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12
VACUUM_PERMEABILITY_H_M = 1.25663706212e-6


class LayeredEarthModel(StrictModule):
    """Horizontal layers over a halfspace under exp(-i omega t)."""

    thickness_m: Array
    horizontal_conductivity_S_m: Array
    vertical_conductivity_S_m: Array
    horizontal_permittivity_F_m: Array
    vertical_permittivity_F_m: Array
    permeability_H_m: Array
    layer_count: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        thickness_m: ArrayLike,
        horizontal_conductivity_S_m: ArrayLike,
        /,
        *,
        vertical_conductivity_S_m: ArrayLike | None = None,
        horizontal_permittivity_F_m: ArrayLike = VACUUM_PERMITTIVITY_F_M,
        vertical_permittivity_F_m: ArrayLike | None = None,
        permeability_H_m: ArrayLike = VACUUM_PERMEABILITY_H_M,
    ):
        thickness = jnp.asarray(thickness_m)
        horizontal = jnp.asarray(horizontal_conductivity_S_m)
        if thickness.ndim != 1 or horizontal.shape != (thickness.size + 1,):
            raise ValueError(
                "Layered earth requires n thicknesses and n+1 material values."
            )
        vertical = (
            horizontal
            if vertical_conductivity_S_m is None
            else jnp.asarray(vertical_conductivity_S_m)
        )
        epsilon_h = jnp.broadcast_to(
            jnp.asarray(horizontal_permittivity_F_m), horizontal.shape
        )
        epsilon_v = (
            epsilon_h
            if vertical_permittivity_F_m is None
            else jnp.asarray(vertical_permittivity_F_m)
        )
        mu = jnp.broadcast_to(jnp.asarray(permeability_H_m), horizontal.shape)
        if vertical.shape != horizontal.shape or epsilon_v.shape != horizontal.shape:
            raise ValueError("Layered anisotropic electric material shapes disagree.")
        thickness = eqx.error_if(
            thickness,
            jnp.any(~jnp.isfinite(thickness))
            | jnp.any(thickness <= 0)
            | jnp.any(~jnp.isfinite(horizontal))
            | jnp.any(horizontal <= 0)
            | jnp.any(~jnp.isfinite(vertical))
            | jnp.any(vertical <= 0)
            | jnp.any(~jnp.isfinite(epsilon_h))
            | jnp.any(epsilon_h <= 0)
            | jnp.any(~jnp.isfinite(epsilon_v))
            | jnp.any(epsilon_v <= 0)
            | jnp.any(~jnp.isfinite(mu))
            | jnp.any(mu <= 0),
            "Layered earth thickness and passive material values must be finite and positive.",
        )
        self.thickness_m = thickness
        self.horizontal_conductivity_S_m = horizontal
        self.vertical_conductivity_S_m = vertical
        self.horizontal_permittivity_F_m = epsilon_h
        self.vertical_permittivity_F_m = epsilon_v
        self.permeability_H_m = mu
        self.layer_count = int(horizontal.size)
        self.model_id = canonical_fingerprint(
            {
                "kind": "layered-earth-em",
                "thickness_m": np.asarray(thickness),
                "horizontal_conductivity_S_m": np.asarray(horizontal),
                "vertical_conductivity_S_m": np.asarray(vertical),
                "horizontal_permittivity_F_m": np.asarray(epsilon_h),
                "vertical_permittivity_F_m": np.asarray(epsilon_v),
                "permeability_H_m": np.asarray(mu),
                "convention": "exp(-i-omega-t)",
            }
        )

    def propagation(
        self,
        angular_frequency: ArrayLike,
        horizontal_wavenumber_m_inverse: ArrayLike,
        polarization: Literal["te", "tm"],
        /,
    ) -> tuple[Array, Array]:
        omega = jnp.asarray(angular_frequency)
        wave = jnp.asarray(horizontal_wavenumber_m_inverse)
        if polarization not in ("te", "tm"):
            raise ValueError("Layered polarization must be TE or TM.")
        omega = eqx.error_if(
            omega,
            jnp.any(~jnp.isfinite(omega))
            | jnp.any(omega <= 0)
            | jnp.any(~jnp.isfinite(wave))
            | jnp.any(wave < 0),
            "Layered frequency and horizontal wavenumber must be finite and physical.",
        )
        admittance_h = (
            self.horizontal_conductivity_S_m
            - 1j * omega[..., None] * self.horizontal_permittivity_F_m
        )
        admittance_v = (
            self.vertical_conductivity_S_m
            - 1j * omega[..., None] * self.vertical_permittivity_F_m
        )
        wave_squared = wave[..., None] ** 2
        if polarization == "te":
            propagation = jnp.sqrt(
                wave_squared
                - 1j * omega[..., None] * self.permeability_H_m * admittance_h
            )
            impedance = -1j * omega[..., None] * self.permeability_H_m / propagation
        else:
            propagation = jnp.sqrt(
                (admittance_h / admittance_v) * wave_squared
                - 1j * omega[..., None] * self.permeability_H_m * admittance_h
            )
            impedance = propagation / admittance_h
        propagation = jnp.where(jnp.real(propagation) < 0, -propagation, propagation)
        return propagation, impedance

    def surface_impedance(
        self,
        angular_frequency: ArrayLike,
        horizontal_wavenumber_m_inverse: ArrayLike = 0.0,
        /,
        *,
        polarization: Literal["te", "tm"] = "te",
    ) -> Array:
        propagation, intrinsic = self.propagation(
            angular_frequency, horizontal_wavenumber_m_inverse, polarization
        )
        impedance = intrinsic[..., -1]
        for layer in range(self.layer_count - 2, -1, -1):
            tangent = jnp.tanh(propagation[..., layer] * self.thickness_m[layer])
            local = intrinsic[..., layer]
            impedance = (
                local * (impedance + local * tangent) / (local + impedance * tangent)
            )
        return impedance

    def magnetotelluric_impedance(self, frequencies_Hz: ArrayLike, /) -> Array:
        frequency = jnp.asarray(frequencies_Hz)
        return self.surface_impedance(2 * jnp.pi * frequency, 0.0, polarization="te")


class DigitalHankelTransformPlan(StrictModule, NonTrainableState):
    wavenumbers_m_inverse: Array
    weights_m_inverse: Array
    order: Literal[0, 1] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        wavenumbers_m_inverse: ArrayLike,
        weights_m_inverse: ArrayLike,
        order: Literal[0, 1],
        /,
    ):
        wave = np.asarray(wavenumbers_m_inverse, dtype=float)
        weights = np.asarray(weights_m_inverse, dtype=float)
        if (
            wave.ndim != 1
            or wave.size < 4
            or weights.shape != wave.shape
            or np.any(~np.isfinite(wave))
            or np.any(wave <= 0)
            or np.any(np.diff(wave) <= 0)
            or np.any(~np.isfinite(weights))
            or order not in (0, 1)
        ):
            raise ValueError("Hankel wavenumbers/weights/order are invalid.")
        self.wavenumbers_m_inverse, self.weights_m_inverse = (
            jnp.asarray(wave),
            jnp.asarray(weights),
        )
        self.order = order
        self.plan_id = canonical_fingerprint(
            {
                "kind": "digital-hankel-transform",
                "wave": wave,
                "weights": weights,
                "order": order,
            }
        )

    @classmethod
    def gauss_legendre(
        cls,
        maximum_wavenumber_m_inverse: float,
        count: int,
        order: Literal[0, 1],
        /,
    ) -> DigitalHankelTransformPlan:
        maximum, count_ = float(maximum_wavenumber_m_inverse), int(count)
        if not np.isfinite(maximum) or maximum <= 0 or count_ < 4:
            raise ValueError("Hankel maximum/count must be positive and count >=4.")
        nodes, weights = np.polynomial.legendre.leggauss(count_)
        return cls(0.5 * maximum * (nodes + 1), 0.5 * maximum * weights, order)

    def evaluate(self, kernel: ArrayLike, offsets_m: ArrayLike, /) -> Array:
        values = jnp.asarray(kernel)
        offsets = jnp.asarray(offsets_m)
        if values.shape[-1] != self.wavenumbers_m_inverse.size or offsets.ndim != 1:
            raise ValueError("Hankel kernel or offset shape is invalid.")
        argument = offsets[:, None] * self.wavenumbers_m_inverse[None, :]
        bessel = jv(self.order, argument)
        return jnp.sum(
            values[..., None, :]
            * bessel.reshape((1,) * (values.ndim - 1) + bessel.shape)
            * self.weights_m_inverse,
            axis=-1,
        )


__all__ = [
    "DigitalHankelTransformPlan",
    "LayeredEarthModel",
    "VACUUM_PERMEABILITY_H_M",
    "VACUUM_PERMITTIVITY_F_M",
]
