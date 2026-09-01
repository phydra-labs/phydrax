#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class HaloMassFunctionPlan(StrictModule, NonTrainableState):
    collapse_threshold: Array
    amplitude: Array
    exponent: Array

    def __init__(self, collapse_threshold=1.686, amplitude=0.3222, exponent=0.3, /):
        self.collapse_threshold = jnp.asarray(collapse_threshold).reshape(())
        self.amplitude = jnp.asarray(amplitude).reshape(())
        self.exponent = jnp.asarray(exponent).reshape(())

    def multiplicity(self, variance: ArrayLike, /) -> Array:
        sigma = jnp.asarray(variance)
        nu = self.collapse_threshold / sigma
        return (
            self.amplitude
            * jnp.sqrt(2.0 / jnp.pi)
            * nu
            * (1.0 + nu ** (-2.0 * self.exponent))
            * jnp.exp(-0.5 * nu**2)
        )


class HaloModelResult(StrictModule):
    one_halo: Array
    two_halo: Array
    total: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


class HaloModelPlan(StrictModule, NonTrainableState):
    mass: Array
    mass_function: Array
    bias: Array
    profile_fourier: Array
    mean_density: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mass,
        mass_function,
        bias,
        profile_fourier,
        mean_density,
        /,
        *,
        model_id="halo-model",
    ):
        self.mass = jnp.asarray(mass)
        self.mass_function = jnp.asarray(mass_function)
        self.bias = jnp.asarray(bias)
        self.profile_fourier = jnp.asarray(profile_fourier)
        self.mean_density = jnp.asarray(mean_density).reshape(())
        if (
            self.profile_fourier.ndim != 2
            or self.profile_fourier.shape[0] != self.mass.size
        ):
            raise ValueError("Halo profile Fourier table has incompatible shape.")
        self.plan_id = canonical_fingerprint(
            {
                "kind": "halo-model-plan",
                "model_id": str(model_id),
                "masses": int(self.mass.size),
            }
        )

    def evaluate(self, linear_power: ArrayLike, /) -> HaloModelResult:
        linear = jnp.asarray(linear_power)
        spacing = self.mass[1:] - self.mass[:-1]
        quadrature_width = jnp.concatenate(
            (
                0.5 * spacing[:1],
                0.5 * (spacing[:-1] + spacing[1:]),
                0.5 * spacing[-1:],
            )
        )
        weights = quadrature_width * self.mass_function
        one = jnp.sum(
            weights[:, None]
            * (self.mass[:, None] / self.mean_density) ** 2
            * self.profile_fourier**2,
            axis=0,
        )
        bias_integral = jnp.sum(
            weights[:, None]
            * self.bias[:, None]
            * self.mass[:, None]
            / self.mean_density
            * self.profile_fourier,
            axis=0,
        )
        two = bias_integral**2 * linear
        valid = jnp.all(jnp.isfinite(one + two)) & (self.mean_density > 0.0)
        return HaloModelResult(one, two, one + two, valid, self.plan_id)


class CmbLensingPlan(StrictModule, NonTrainableState):
    multipoles: Array
    deflection_variance: Array

    def __init__(self, multipoles, deflection_variance, /):
        self.multipoles = jnp.asarray(multipoles)
        self.deflection_variance = jnp.asarray(deflection_variance).reshape(())

    def lens(self, unlensed_cl: ArrayLike, /) -> Array:
        unlensed = jnp.asarray(unlensed_cl)
        damping = jnp.exp(
            -0.5 * self.multipoles * (self.multipoles + 1.0) * self.deflection_variance
        )
        return damping * unlensed + (1.0 - damping) * jnp.interp(
            self.multipoles, self.multipoles, unlensed
        )


class LightConeResult(StrictModule):
    positions: Array
    shell_index: Array
    valid: Array
    overflow: Array


class LightConePlan(StrictModule, NonTrainableState):
    shell_radii: Array
    capacity: int = eqx.field(static=True)

    def __init__(self, shell_radii, capacity: int, /):
        self.shell_radii = jnp.asarray(shell_radii)
        self.capacity = int(capacity)

    def select(self, positions: ArrayLike, /) -> LightConeResult:
        values = jnp.asarray(positions)
        radius = jnp.sqrt(jnp.sum(values * values, axis=-1))
        shell = jnp.searchsorted(self.shell_radii, radius)
        active = shell < self.shell_radii.size
        order = jnp.argsort(~active)
        selected = values[order[: self.capacity]]
        selected_shell = shell[order[: self.capacity]]
        count = jnp.sum(active.astype(jnp.int32))
        return LightConeResult(
            selected,
            selected_shell,
            jnp.all(jnp.isfinite(selected)),
            count > self.capacity,
        )


class LensingPlanePlan(StrictModule, NonTrainableState):
    pixel_scale: Array

    def __init__(self, pixel_scale, /):
        self.pixel_scale = jnp.asarray(pixel_scale).reshape(())

    def convergence_and_shear(
        self, surface_density: ArrayLike, critical_density: ArrayLike, /
    ):
        density = jnp.asarray(surface_density)
        convergence = density / jnp.asarray(critical_density)
        ny, nx = density.shape
        ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=self.pixel_scale)
        kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=self.pixel_scale)
        kxx, kyy = jnp.meshgrid(kx, ky)
        squared = kxx**2 + kyy**2
        transformed = jnp.fft.fft2(convergence)
        gamma1 = jnp.fft.ifft2(
            jnp.where(squared > 0.0, (kxx**2 - kyy**2) / squared * transformed, 0.0)
        ).real
        gamma2 = jnp.fft.ifft2(
            jnp.where(squared > 0.0, 2.0 * kxx * kyy / squared * transformed, 0.0)
        ).real
        return convergence, gamma1, gamma2


class BaryonicFeedbackPlan(StrictModule, NonTrainableState):
    amplitude: Array
    pivot_wavenumber: Array
    slope: Array

    def __init__(self, amplitude, pivot_wavenumber, slope, /):
        self.amplitude = jnp.asarray(amplitude).reshape(())
        self.pivot_wavenumber = jnp.asarray(pivot_wavenumber).reshape(())
        self.slope = jnp.asarray(slope).reshape(())

    def apply(self, wavenumber: ArrayLike, power: ArrayLike, /) -> Array:
        k = jnp.asarray(wavenumber)
        suppression = 1.0 - self.amplitude * (k / self.pivot_wavenumber) ** self.slope / (
            1.0 + (k / self.pivot_wavenumber) ** self.slope
        )
        return jnp.asarray(power) * suppression


__all__ = [
    "BaryonicFeedbackPlan",
    "CmbLensingPlan",
    "HaloMassFunctionPlan",
    "HaloModelPlan",
    "HaloModelResult",
    "LensingPlanePlan",
    "LightConePlan",
    "LightConeResult",
]
