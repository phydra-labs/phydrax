#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Thermal electron–ion free–free (bremsstrahlung) spectral coefficients.

For electrons of density ``n_e`` and temperature ``T`` scattering on ions of
density ``n_i`` and charge ``Z e`` (Rybicki & Lightman 1979, eq. 5.14a, written
with ``e² → e²/(4πε₀)``)::

    ε_ν = (32π e⁶ / (3 m c³ (4πε₀)³)) √(2π/(3 k T m)) Z² n_e n_i e^{−hν/kT} ḡ

is the power per volume per unit frequency ``ν``. The emission coefficient per
unit angular frequency and steradian is ``j_ω = ε_ν / (8π²)``; absorption follows
Kirchhoff's law with stimulated emission, ``α = j_ν / B_ν(T)`` with
``j_ν = 2π j_ω``. The thermally averaged Gaunt factor is the Born result
``ḡ = (√3/π) e^{u/2} K₀(u/2)``, ``u = hν/kT`` (Karzas & Latter 1961), valid when
``Z² Ry/(kT) ≪ 1``; the nonrelativistic and ``ω > ω_p`` limits are also reported.
Its Planck average is exactly ``2√3/π``.
"""

from __future__ import annotations

from math import isfinite, log, pi, sqrt

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._physical import ElectromagneticScaleContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..special import kve
from ..typing import checked
from ._gray_means import gray_mean_opacities, GrayMeanOpacities


def _float64(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.inexact) and array.dtype != jnp.float64:
        raise TypeError(f"{name} requires float64 values; received {array.dtype}.")
    return array.astype(jnp.float64)


def born_thermal_gaunt(photon_energy_ratio: ArrayLike, /) -> Array:
    """Thermally averaged Born Gaunt factor ``(√3/π) e^{u/2} K₀(u/2)``."""
    u = _float64(photon_energy_ratio, "photon_energy_ratio")
    return (sqrt(3.0) / pi) * kve(0.0, 0.5 * u)


class ThermalFreeFreeEvidence(StrictModule):
    """Per-sample validity of the Born, nonrelativistic, ``ω > ω_p`` free–free form."""

    finite: Array
    physically_valid: Array
    born_valid: Array
    nonrelativistic: Array
    above_plasma_frequency: Array
    qualified: Array


class ThermalFreeFreeCoefficients(StrictModule):
    """Unpolarized free–free transfer coefficients per unit angular frequency.

    ``emission`` is the Stokes vector ``(j_ω, 0, 0, 0)`` per volume, steradian and
    unit angular frequency; ``propagation_matrix`` is ``α 𝟙₄`` per unit length.
    """

    emission: Array
    absorption: Array
    propagation_matrix: Array
    gaunt: Array
    born_parameter: Array
    angular_frequency: Array
    evidence: ThermalFreeFreeEvidence
    model_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)


class ThermalFreeFreeModel(StrictModule, NonTrainableState):
    """Thermal free–free emission and absorption bound to an electromagnetic scale.

    Densities are per cubic length unit, temperatures in kelvin, angular
    frequencies per time unit of ``scale``. ``maximum_born_parameter`` bounds
    ``Z² Ry/(kT)``, ``maximum_temperature_ratio`` bounds ``kT/(m c²)``.
    """

    scale: ElectromagneticScaleContract = eqx.field(static=True)
    ion_charge_number: float = eqx.field(static=True)
    maximum_born_parameter: float = eqx.field(static=True)
    maximum_temperature_ratio: float = eqx.field(static=True)
    log_emission_prefactor: float = eqx.field(static=True)
    rydberg_energy: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        scale: ElectromagneticScaleContract,
        /,
        *,
        ion_charge_number: float = 1.0,
        maximum_born_parameter: float = 0.1,
        maximum_temperature_ratio: float = 0.05,
    ) -> None:
        values = (
            float(ion_charge_number),
            float(maximum_born_parameter),
            float(maximum_temperature_ratio),
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(
                "ion_charge_number and the validity bounds must be finite and positive."
            )
        e = float(scale.elementary_charge)
        m = float(scale.electron_mass)
        c = float(scale.speed_of_light)
        epsilon = float(scale.vacuum_permittivity)
        k = float(scale.relativity.boltzmann_constant)
        # log of 32π e⁶/(3 m c³ (4πε₀)³) · √(2π/(3 k m)) / (8π²): j_ω · √T / (Z² n_e n_i).
        self.log_emission_prefactor = (
            log(32.0 * pi / 3.0)
            + 6.0 * log(e)
            - log(m)
            - 3.0 * log(c)
            - 3.0 * log(4.0 * pi * epsilon)
            + 0.5 * (log(2.0 * pi / 3.0) - log(k) - log(m))
            - log(8.0 * pi * pi)
        )
        self.rydberg_energy = float(scale.fine_structure) ** 2 * m * c * c / 2.0
        self.scale = scale
        (
            self.ion_charge_number,
            self.maximum_born_parameter,
            self.maximum_temperature_ratio,
        ) = values
        self.model_id = canonical_fingerprint(
            {
                "kind": "thermal-free-free",
                "scale": scale.scale_id,
                "gaunt": "born-thermal-average",
                "ion_charge_number": values[0],
                "maximum_born_parameter": values[1],
                "maximum_temperature_ratio": values[2],
            }
        )

    def _constants(self) -> tuple[float, float, float, float]:
        scale = self.scale
        return (
            2.0 * pi * float(scale.reduced_planck_constant),
            float(scale.relativity.boltzmann_constant),
            float(scale.speed_of_light),
            float(scale.electron_mass),
        )

    def _absorption(
        self,
        electron_density: Array,
        ion_density: Array,
        temperature: Array,
        frequency: Array,
    ) -> tuple[Array, Array, Array]:
        """``(j_ω, α, ḡ)`` at frequency ``ν`` (cycles per time unit)."""
        h, k, c, _ = self._constants()
        u = h * frequency / (k * temperature)
        gaunt = born_thermal_gaunt(u)
        z = self.ion_charge_number
        emission = (
            jnp.exp(self.log_emission_prefactor - u)
            * gaunt
            * z
            * z
            * electron_density
            * ion_density
            / jnp.sqrt(temperature)
        )
        # α = 2π j_ω / B_ν with B_ν = 2hν³/(c² (eᵘ − 1)).
        absorption = 2.0 * pi * emission * c * c * jnp.expm1(u) / (2.0 * h * frequency**3)
        return emission, absorption, gaunt

    def _validity(
        self, electron_density: Array, temperature: Array, angular_frequency: Array
    ) -> tuple[Array, Array, Array, Array]:
        _, k, c, m = self._constants()
        scale = self.scale
        z = self.ion_charge_number
        born_parameter = z * z * self.rydberg_energy / (k * temperature)
        plasma_squared = (
            electron_density
            * float(scale.elementary_charge) ** 2
            / (float(scale.vacuum_permittivity) * m)
        )
        return (
            born_parameter,
            born_parameter <= self.maximum_born_parameter,
            k * temperature / (m * c * c) <= self.maximum_temperature_ratio,
            angular_frequency * angular_frequency > plasma_squared,
        )

    def evaluate(
        self,
        electron_density: ArrayLike,
        ion_density: ArrayLike,
        temperature: ArrayLike,
        angular_frequency: ArrayLike,
        /,
    ) -> ThermalFreeFreeCoefficients:
        density, ions, kelvin, omega = jnp.broadcast_arrays(
            _float64(electron_density, "electron_density"),
            _float64(ion_density, "ion_density"),
            _float64(temperature, "temperature"),
            _float64(angular_frequency, "angular_frequency"),
        )
        physical = (
            jnp.isfinite(density)
            & jnp.isfinite(ions)
            & jnp.isfinite(kelvin)
            & jnp.isfinite(omega)
            & (density >= 0.0)
            & (ions >= 0.0)
            & (kelvin > 0.0)
            & (omega > 0.0)
        )
        safe_kelvin = jnp.where(physical, kelvin, 1.0)
        safe_omega = jnp.where(physical, omega, 1.0)
        emission, absorption, gaunt = self._absorption(
            density, ions, safe_kelvin, safe_omega / (2.0 * pi)
        )
        born_parameter, born_valid, nonrelativistic, above = self._validity(
            density, safe_kelvin, safe_omega
        )
        nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
        emission = jnp.where(physical, emission, nan)
        absorption = jnp.where(physical, absorption, nan)
        zero = jnp.zeros_like(emission)
        stokes = jnp.stack((emission, zero, zero, zero), axis=-1)
        propagation = absorption[..., None, None] * jnp.eye(4, dtype=jnp.float64)
        finite = jnp.isfinite(emission) & jnp.isfinite(absorption)
        evidence = ThermalFreeFreeEvidence(
            finite=finite,
            physically_valid=physical,
            born_valid=physical & born_valid,
            nonrelativistic=physical & nonrelativistic,
            above_plasma_frequency=physical & above,
            qualified=finite & physical & born_valid & nonrelativistic & above,
        )
        return ThermalFreeFreeCoefficients(
            emission=stokes,
            absorption=absorption,
            propagation_matrix=propagation,
            gaunt=jnp.where(physical, gaunt, nan),
            born_parameter=jnp.where(physical, born_parameter, nan),
            angular_frequency=omega,
            evidence=evidence,
            model_id=self.model_id,
            scale_id=self.scale.scale_id,
        )

    def gray_means(
        self,
        electron_density: ArrayLike,
        ion_density: ArrayLike,
        temperature: ArrayLike,
        radiation_temperature: ArrayLike,
        /,
        *,
        panels: int = 24,
        order: int = 21,
        tolerance: float = 1.0e-6,
    ) -> GrayMeanOpacities:
        """Planck and Rosseland means over ``hν/kT ∈ [10⁻¹², 60]`` of both temperatures.

        The means use the vacuum Planck function, so they are supported only
        where ``hν_p ≤ 10⁻³ k min(T, T_r)`` (the Planck weight below the plasma
        frequency is then below ``10⁻¹⁰``) and every node lies in the Born and
        nonrelativistic support.
        """
        density, ions, kelvin, radiation = jnp.broadcast_arrays(
            _float64(electron_density, "electron_density"),
            _float64(ion_density, "ion_density"),
            _float64(temperature, "temperature"),
            _float64(radiation_temperature, "radiation_temperature"),
        )
        h, k, c, m = self._constants()
        coldest = jnp.minimum(kelvin, radiation)
        hottest = jnp.maximum(kelvin, radiation)
        plasma_frequency = jnp.sqrt(
            density
            * float(self.scale.elementary_charge) ** 2
            / (float(self.scale.vacuum_permittivity) * m)
        ) / (2.0 * pi)
        transparent = h * plasma_frequency <= 1.0e-3 * k * coldest

        def absorption(frequency: Array) -> tuple[Array, Array]:
            _, alpha, _ = self._absorption(
                density[..., None], ions[..., None], kelvin[..., None], frequency
            )
            _, born_valid, nonrelativistic, _ = self._validity(
                density[..., None], kelvin[..., None], 2.0 * pi * frequency
            )
            return alpha, born_valid & nonrelativistic & transparent[..., None]

        return gray_mean_opacities(
            absorption,
            kelvin,
            radiation,
            1.0e-12 * k * coldest / h,
            60.0 * k * hottest / h,
            planck_constant=h,
            boltzmann_constant=k,
            speed_of_light=c,
            panels=panels,
            order=order,
            tolerance=tolerance,
        )


__all__ = [
    "born_thermal_gaunt",
    "ThermalFreeFreeCoefficients",
    "ThermalFreeFreeEvidence",
    "ThermalFreeFreeModel",
]
