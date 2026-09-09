#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measure-neutral jump and Lévy model records."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ._diffusion import HestonModel


def _scalar(value: ArrayLike, name: str, /, *, positive: bool = False) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    invalid = ~jnp.isfinite(array) | ((array <= 0.0) if positive else False)
    return eqx.error_if(
        array, invalid, f"{name} must be finite" + (" and positive." if positive else ".")
    )


class MertonJumpDiffusionModel(StrictModule):
    """Lognormal compound-Poisson jumps plus constant diffusion."""

    diffusion_volatility: Array
    jump_intensity: Array
    jump_mean: Array
    jump_volatility: Array

    def __init__(
        self,
        diffusion_volatility: ArrayLike,
        jump_intensity: ArrayLike,
        jump_mean: ArrayLike,
        jump_volatility: ArrayLike,
        /,
    ):
        self.diffusion_volatility = _scalar(
            diffusion_volatility, "diffusion_volatility", positive=True
        )
        self.jump_intensity = _scalar(jump_intensity, "jump_intensity", positive=True)
        self.jump_mean = _scalar(jump_mean, "jump_mean")
        self.jump_volatility = _scalar(jump_volatility, "jump_volatility", positive=True)

    @property
    def exponential_compensator(self) -> Array:
        return self.jump_intensity * (
            jnp.exp(self.jump_mean + 0.5 * self.jump_volatility**2) - 1.0
        )

    def characteristic_exponent(self, frequency: ArrayLike, /) -> Array:
        u = jnp.asarray(frequency, dtype=complex)
        jump = jnp.exp(1j * u * self.jump_mean - 0.5 * self.jump_volatility**2 * u**2)
        return (
            -0.5 * self.diffusion_volatility**2 * (u**2 + 1j * u)
            + self.jump_intensity * (jump - 1.0)
            - 1j * u * self.exponential_compensator
        )


class KouJumpDiffusionModel(StrictModule):
    """Double-exponential compound-Poisson jumps and constant diffusion."""

    diffusion_volatility: Array
    jump_intensity: Array
    upward_probability: Array
    upward_rate: Array
    downward_rate: Array

    def __init__(
        self,
        diffusion_volatility: ArrayLike,
        jump_intensity: ArrayLike,
        upward_probability: ArrayLike,
        upward_rate: ArrayLike,
        downward_rate: ArrayLike,
        /,
    ):
        probability = _scalar(upward_probability, "upward_probability")
        probability = eqx.error_if(
            probability,
            (probability <= 0.0) | (probability >= 1.0),
            "upward_probability must lie strictly in (0, 1).",
        )
        upward = _scalar(upward_rate, "upward_rate", positive=True)
        upward = eqx.error_if(
            upward,
            upward <= 1.0,
            "upward_rate must exceed one so the asset first moment exists.",
        )
        self.diffusion_volatility = _scalar(
            diffusion_volatility, "diffusion_volatility", positive=True
        )
        self.jump_intensity = _scalar(jump_intensity, "jump_intensity", positive=True)
        self.upward_probability = probability
        self.upward_rate = upward
        self.downward_rate = _scalar(downward_rate, "downward_rate", positive=True)

    @property
    def exponential_compensator(self) -> Array:
        expected_multiplier = self.upward_probability * self.upward_rate / (
            self.upward_rate - 1.0
        ) + (1.0 - self.upward_probability) * self.downward_rate / (
            self.downward_rate + 1.0
        )
        return self.jump_intensity * (expected_multiplier - 1.0)

    def characteristic_exponent(self, frequency: ArrayLike, /) -> Array:
        u = jnp.asarray(frequency, dtype=complex)
        jump_cf = self.upward_probability * self.upward_rate / (
            self.upward_rate - 1j * u
        ) + (1.0 - self.upward_probability) * self.downward_rate / (
            self.downward_rate + 1j * u
        )
        return (
            -0.5 * self.diffusion_volatility**2 * (u**2 + 1j * u)
            + self.jump_intensity * (jump_cf - 1.0)
            - 1j * u * self.exponential_compensator
        )


class VarianceGammaModel(StrictModule):
    """Variance-gamma Lévy exponent with a finite exponential first moment."""

    volatility: Array
    skew: Array
    variance_rate: Array

    def __init__(
        self, volatility: ArrayLike, skew: ArrayLike, variance_rate: ArrayLike, /
    ):
        volatility = _scalar(volatility, "volatility", positive=True)
        skew = _scalar(skew, "skew")
        rate = _scalar(variance_rate, "variance_rate", positive=True)
        rate = eqx.error_if(
            rate,
            1.0 - skew * rate - 0.5 * volatility**2 * rate <= 0.0,
            "Variance-gamma parameters do not admit the asset first moment.",
        )
        self.volatility = volatility
        self.skew = skew
        self.variance_rate = rate

    @property
    def exponential_compensator(self) -> Array:
        return (
            jnp.log(
                1.0
                - self.skew * self.variance_rate
                - 0.5 * self.volatility**2 * self.variance_rate
            )
            / self.variance_rate
        )

    def characteristic_exponent(self, frequency: ArrayLike, /) -> Array:
        u = jnp.asarray(frequency, dtype=complex)
        raw = (
            -jnp.log(
                1.0
                - 1j * self.skew * self.variance_rate * u
                + 0.5 * self.volatility**2 * self.variance_rate * u**2
            )
            / self.variance_rate
        )
        return raw - 1j * u * (-self.exponential_compensator)


class NormalInverseGaussianModel(StrictModule):
    """Normal-inverse-Gaussian Lévy exponent with admissible first moment."""

    tail: Array
    skew: Array
    scale: Array

    def __init__(self, tail: ArrayLike, skew: ArrayLike, scale: ArrayLike, /):
        tail = _scalar(tail, "tail", positive=True)
        skew = _scalar(skew, "skew")
        tail = eqx.error_if(
            tail,
            tail <= jnp.abs(skew + 1.0),
            "NIG tail must exceed abs(skew + 1) so the asset first moment exists.",
        )
        self.tail = tail
        self.skew = skew
        self.scale = _scalar(scale, "scale", positive=True)

    @property
    def exponential_compensator(self) -> Array:
        return self.scale * (
            jnp.sqrt(self.tail**2 - self.skew**2)
            - jnp.sqrt(self.tail**2 - (self.skew + 1.0) ** 2)
        )

    def characteristic_exponent(self, frequency: ArrayLike, /) -> Array:
        u = jnp.asarray(frequency, dtype=complex)
        raw = self.scale * (
            jnp.sqrt(self.tail**2 - self.skew**2)
            - jnp.sqrt(self.tail**2 - (self.skew + 1j * u) ** 2)
        )
        return raw - 1j * u * self.exponential_compensator


class CGMYModel(StrictModule):
    """CGMY tempered-stable model for non-integer activity in (0, 2)."""

    scale: Array
    left_rate: Array
    right_rate: Array
    activity: Array

    def __init__(
        self,
        scale: ArrayLike,
        left_rate: ArrayLike,
        right_rate: ArrayLike,
        activity: ArrayLike,
        /,
    ):
        activity_ = _scalar(activity, "activity")
        activity_ = eqx.error_if(
            activity_,
            (activity_ <= 0.0) | (activity_ >= 2.0) | jnp.isclose(activity_, 1.0),
            "This CGMY exponent supports activity in (0, 2) excluding the singular Y=1 parameterization.",
        )
        right = _scalar(right_rate, "right_rate", positive=True)
        right = eqx.error_if(
            right,
            right <= 1.0,
            "right_rate must exceed one so the asset first moment exists.",
        )
        self.scale = _scalar(scale, "scale", positive=True)
        self.left_rate = _scalar(left_rate, "left_rate", positive=True)
        self.right_rate = right
        self.activity = activity_

    def _raw_exponent(self, frequency: Array) -> Array:
        u = jnp.asarray(frequency, dtype=complex)
        return (
            self.scale
            * jsp.special.gamma(-self.activity)
            * (
                (self.right_rate - 1j * u) ** self.activity
                - self.right_rate**self.activity
                + (self.left_rate + 1j * u) ** self.activity
                - self.left_rate**self.activity
            )
        )

    @property
    def exponential_compensator(self) -> Array:
        return jnp.real(self._raw_exponent(jnp.asarray(-1j)))

    def characteristic_exponent(self, frequency: ArrayLike, /) -> Array:
        u = jnp.asarray(frequency, dtype=complex)
        return self._raw_exponent(u) - 1j * u * self.exponential_compensator


class BatesModel(StrictModule):
    """Heston variance dynamics with independent lognormal compound-Poisson jumps."""

    heston: HestonModel
    jump_intensity: Array
    jump_mean: Array
    jump_volatility: Array

    def __init__(
        self,
        heston: HestonModel,
        jump_intensity: ArrayLike,
        jump_mean: ArrayLike,
        jump_volatility: ArrayLike,
        /,
    ):
        if not isinstance(heston, HestonModel):
            raise TypeError("heston must be a HestonModel.")
        self.heston = heston
        self.jump_intensity = _scalar(jump_intensity, "jump_intensity", positive=True)
        self.jump_mean = _scalar(jump_mean, "jump_mean")
        self.jump_volatility = _scalar(jump_volatility, "jump_volatility", positive=True)

    @property
    def exponential_compensator(self) -> Array:
        return self.jump_intensity * (
            jnp.exp(self.jump_mean + 0.5 * self.jump_volatility**2) - 1.0
        )

    def jump_characteristic_exponent(self, frequency: ArrayLike, /) -> Array:
        u = jnp.asarray(frequency, dtype=complex)
        jump = jnp.exp(1j * u * self.jump_mean - 0.5 * self.jump_volatility**2 * u**2)
        return self.jump_intensity * (jump - 1.0) - 1j * u * self.exponential_compensator


__all__ = [
    "BatesModel",
    "CGMYModel",
    "KouJumpDiffusionModel",
    "MertonJumpDiffusionModel",
    "NormalInverseGaussianModel",
    "VarianceGammaModel",
]
