#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Waterside oxide, hydrogen pickup, and surface emissivity of Zircaloy-4 cladding.

Every law evaluates its published correlation in the source variables and
converts at the boundary to SI: K, m, s, W, kg; fast neutron flux in
n/(m^2 s) (E > 1 MeV). Sources:

- MATPRO: Siefken et al., "SCDAP/RELAP5/MOD 3.3 Code Manual: MATPRO - A Library
  of Materials Properties for Light-Water-Reactor Accident Analysis",
  NUREG/CR-6150 Rev. 2, Vol. 4 (INEL-96/0422), 2001.
- PNNL-19417: Luscher, Geelhood and Porter, "Material Property Correlations:
  Comparisons between FRAPCON-4.0, FRAPTRAN-2.0, and MATPRO", PNNL-19417
  Rev. 2 (NUREG/CR-7024 Rev. 2), 2015, including its Appendix A source listings.
- FRAPCON-4.0: Geelhood, Luscher, Raynaud and Porter, "FRAPCON-4.0: A Computer
  Code for the Calculation of Steady-State, Thermal-Mechanical Behavior of Oxide
  Fuel Rods for High Burnup", PNNL-19418 Vol. 1 Rev. 2 (NUREG/CR-7022 Vol. 1
  Rev. 2), 2015.

Out-of-range evaluation returns the formula value; each ``validity`` states the
fitted domain and the caller decides.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...materials import PropertyValidityRange
from ._cladding_properties import ZIRCALOY_DENSITY
from ._constants import SECONDS_PER_DAY
from ._identity import CladdingAlloy
from ._material_math import cubic_polynomial, zero_safe_power
from ._material_parameters import (
    finite_parameter,
    float64_array,
    parameter_arrays,
    positive_parameter,
)


_SQUARE_CENTIMETERS_PER_SQUARE_METER = 1.0e4
_MICROMETERS_PER_METER = 1.0e6
_METERS_PER_CENTIMETER = 1.0e-2
_KILOGRAMS_PER_CUBIC_METER_PER_GRAM_PER_CUBIC_CENTIMETER = 1.0e3
_PARTS_PER_MILLION = 1.0e6

# IUPAC conventional standard atomic weights [g/mol].
_HYDROGEN_MOLAR_MASS = 1.008
_OXYGEN_MOLAR_MASS = 15.999
_ZIRCONIUM_MOLAR_MASS = 91.224

# FRAPCON-4.0 Section 2.6.1: gas constant used by the corrosion law [cal/(mol K)].
_CORROSION_GAS_CONSTANT = 1.98


@final
class ZircaloyWatersideCorrosion(StrictModule):
    """Waterside oxidation of Zircaloy-4 in a PWR (FRAPCON-4.0 CORROS).

    EPRI/C-E/KWU model (Garzarolli et al., EPRI NP-2789, 1982) as given in
    FRAPCON-4.0 Section 2.6.1, driven by the metal-oxide interface temperature
    ``T1`` [K]. Pre-transition, below 2.0 um, ``ds/dt = (A / s^2) exp(-Q1 /
    (R T1))`` (Eq. (2.190)), ``A = 6.3e9 um^3/day`` (the unit is printed as
    m^3/day in Eq. (2.191); MATPRO Appendix Eq. (A2-18) gives um^3/day),
    ``Q1 = 32289 cal/mol``, ``R = 1.98 cal/(mol K)``. Equivalently ``d(s^3)/dt
    = 3 A exp(-Q1/(R T1))``, which is regular at ``s = 0``. Post-transition
    the linear law Eq. (2.192) in weight-gain form ``d w / dt = k0 exp(-Q2 /
    (R T1))`` with ``k0 = 11863 + 3.5e4 (1.91e-15 phi)^0.24 g/(cm^2 day)``,
    ``phi`` in n/(cm^2 s), ``Q2 = 27354 cal/mol``, and thickness ``s = w
    gamma`` with ``gamma = 0.6789 cm^3/g`` (Eq. (2.194)). FRAPCON's
    Garzarolli step integral (Eq. (2.193)) is the time integral of this rate
    with ``T1 = T0 + q'' s / lambda``; here the heat flux enters only through
    the caller's interface temperature.

    ``gamma`` fixes the oxygen areal mass per oxide thickness; the metal
    consumed follows the MATPRO mass balance (NUREG/CR-6150 Rev. 2 Vol. 4, Eq.
    (4-318)) with stoichiometric ZrO2 and the metal density.
    """

    pre_transition_rate_constant: Array
    pre_transition_activation_energy: Array
    transition_thickness: Array
    post_transition_base_rate: Array
    post_transition_flux_rate: Array
    flux_scale: Array
    flux_exponent: Array
    post_transition_activation_energy: Array
    oxide_volume_per_oxygen_mass: Array
    metal_density: Array
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    alloy: CladdingAlloy = eqx.field(static=True, default="zircaloy-4")

    def __init__(
        self,
        /,
        *,
        pre_transition_rate_constant: ArrayLike = 6.3e9,
        pre_transition_activation_energy: ArrayLike = 32289.0,
        transition_thickness: ArrayLike = 2.0e-6,
        post_transition_base_rate: ArrayLike = 11863.0,
        post_transition_flux_rate: ArrayLike = 3.5e4,
        flux_scale: ArrayLike = 1.91e-15,
        flux_exponent: ArrayLike = 0.24,
        post_transition_activation_energy: ArrayLike = 27354.0,
        oxide_volume_per_oxygen_mass: ArrayLike = 0.6789,
        metal_density: ArrayLike = ZIRCALOY_DENSITY,
    ) -> None:
        parameters = {
            "pre_transition_rate_constant": positive_parameter(
                pre_transition_rate_constant, "pre_transition_rate_constant"
            ),
            "pre_transition_activation_energy": positive_parameter(
                pre_transition_activation_energy, "pre_transition_activation_energy"
            ),
            "transition_thickness": positive_parameter(
                transition_thickness, "transition_thickness"
            ),
            "post_transition_base_rate": positive_parameter(
                post_transition_base_rate, "post_transition_base_rate"
            ),
            "post_transition_flux_rate": finite_parameter(
                post_transition_flux_rate, "post_transition_flux_rate"
            ),
            "flux_scale": positive_parameter(flux_scale, "flux_scale"),
            "flux_exponent": positive_parameter(flux_exponent, "flux_exponent"),
            "post_transition_activation_energy": positive_parameter(
                post_transition_activation_energy, "post_transition_activation_energy"
            ),
            "oxide_volume_per_oxygen_mass": positive_parameter(
                oxide_volume_per_oxygen_mass, "oxide_volume_per_oxygen_mass"
            ),
            "metal_density": positive_parameter(metal_density, "metal_density"),
        }
        arrays = parameter_arrays(parameters)
        self.pre_transition_rate_constant = arrays["pre_transition_rate_constant"]
        self.pre_transition_activation_energy = arrays["pre_transition_activation_energy"]
        self.transition_thickness = arrays["transition_thickness"]
        self.post_transition_base_rate = arrays["post_transition_base_rate"]
        self.post_transition_flux_rate = arrays["post_transition_flux_rate"]
        self.flux_scale = arrays["flux_scale"]
        self.flux_exponent = arrays["flux_exponent"]
        self.post_transition_activation_energy = arrays[
            "post_transition_activation_energy"
        ]
        self.oxide_volume_per_oxygen_mass = arrays["oxide_volume_per_oxygen_mass"]
        self.metal_density = arrays["metal_density"]
        self.source_id = "frapcon-4.0-pnnl-19418-vol1-rev2:section-2.6.1"
        # MATPRO CORROS (NUREG/CR-6150 Rev. 2 Vol. 4, Section 4.15.1.1 and
        # Appendix A2.2.13): typical reactor operation, 523-673 K.
        self.validity = PropertyValidityRange(
            {"interface_temperature": (523.0, 673.0, "K")}, source_id=self.source_id
        )
        self.law_id = canonical_fingerprint(
            {"kind": "zircaloy-4-pwr-waterside-corrosion", "parameters": parameters}
        )

    def oxide_thickness_rate(
        self,
        oxide_thickness: ArrayLike,
        interface_temperature: ArrayLike,
        fast_flux: ArrayLike,
        /,
    ) -> Array:
        """Oxide growth rate [m/s] at the metal-oxide interface temperature [K].

        ``oxide_thickness`` [m]; ``fast_flux`` [n/(m^2 s)] (E > 1 MeV).
        """

        thickness, t, flux = (
            float64_array(oxide_thickness),
            float64_array(interface_temperature),
            float64_array(fast_flux),
        )
        thickness_um = thickness * _MICROMETERS_PER_METER
        pre_um_per_day = (
            self.pre_transition_rate_constant
            / thickness_um**2
            * jnp.exp(
                -self.pre_transition_activation_energy / (_CORROSION_GAS_CONSTANT * t)
            )
        )
        flux_cm = flux / _SQUARE_CENTIMETERS_PER_SQUARE_METER
        weight_gain_rate = (
            self.post_transition_base_rate
            + self.post_transition_flux_rate
            * zero_safe_power(self.flux_scale * flux_cm, self.flux_exponent)
        ) * jnp.exp(
            -self.post_transition_activation_energy / (_CORROSION_GAS_CONSTANT * t)
        )
        post_m_per_day = (
            weight_gain_rate * self.oxide_volume_per_oxygen_mass * _METERS_PER_CENTIMETER
        )
        return (
            jnp.where(
                thickness < self.transition_thickness,
                pre_um_per_day / _MICROMETERS_PER_METER,
                post_m_per_day,
            )
            / SECONDS_PER_DAY
        )

    def advance(
        self,
        oxide_thickness: ArrayLike,
        interface_temperature: ArrayLike,
        fast_flux: ArrayLike,
        time_step: ArrayLike,
        /,
    ) -> Array:
        """Oxide thickness [m] after a frozen-interface-temperature step [s].

        The cubic pre-transition integral is regular for initially bare metal.
        A step crossing 2 µm spends only its remaining duration in the linear
        post-transition regime. Interface temperature includes the oxide heat
        drop supplied by the coupled thermal residual.
        """
        thickness, temperature, flux, dt = jnp.broadcast_arrays(
            float64_array(oxide_thickness),
            float64_array(interface_temperature),
            float64_array(fast_flux),
            float64_array(time_step),
        )
        cubic_rate = (
            3.0
            * self.pre_transition_rate_constant
            * jnp.exp(
                -self.pre_transition_activation_energy
                / (_CORROSION_GAS_CONSTANT * temperature)
            )
            / (_MICROMETERS_PER_METER**3 * SECONDS_PER_DAY)
        )
        before = thickness < self.transition_thickness
        to_transition = jnp.where(
            before, (self.transition_thickness**3 - thickness**3) / cubic_rate, 0.0
        )
        pre_duration = jnp.minimum(dt, to_transition)
        pre_final = zero_safe_power(thickness**3 + cubic_rate * pre_duration, 1.0 / 3.0)
        post_rate = self.oxide_thickness_rate(
            jnp.broadcast_to(self.transition_thickness, thickness.shape),
            temperature,
            flux,
        )
        post_start = jnp.where(before, self.transition_thickness, thickness)
        post_final = post_start + post_rate * (dt - pre_duration)
        return jnp.where(before & (dt <= to_transition), pre_final, post_final)

    def oxygen_areal_mass(self, oxide_thickness: ArrayLike, /) -> Array:
        """Oxygen taken up [kg/m^2] by an oxide layer (or increment) of thickness [m]."""

        return (
            float64_array(oxide_thickness)
            * _KILOGRAMS_PER_CUBIC_METER_PER_GRAM_PER_CUBIC_CENTIMETER
            / self.oxide_volume_per_oxygen_mass
        )

    def metal_thickness_consumed(self, oxide_thickness: ArrayLike, /) -> Array:
        """Metal thickness [m] converted into an oxide of thickness [m].

        Zirconium mass in the oxide is the oxygen mass times ``M_Zr / (2
        M_O)``; its reciprocal ratio to the oxide thickness is the
        Pilling-Bedworth ratio.
        """

        zirconium = self.oxygen_areal_mass(oxide_thickness) * (
            _ZIRCONIUM_MOLAR_MASS / (2.0 * _OXYGEN_MOLAR_MASS)
        )
        return zirconium / self.metal_density


@final
class ZirconiaThermalConductivity(StrictModule):
    """Thermal conductivity [W/(m K)] of the cladding oxide (FRAPCON-4.0 ZOTCON).

    ``k = 1.9599 - 2.41e-4 T + 6.43e-7 T^2 - 1.946e-10 T^3``, the fit to
    Kingery et al. (J. Am. Ceram. Soc. 37, 1954) adopted from MATPRO-11 Rev. 2
    (PNNL-19417 Rev. 2 Eq. (3.7) and Appendix A.2.3), applicable 375-1673 K.
    """

    constant: Array
    linear: Array
    quadratic: Array
    cubic: Array
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        constant: ArrayLike = 1.9599,
        linear: ArrayLike = -2.41e-4,
        quadratic: ArrayLike = 6.43e-7,
        cubic: ArrayLike = -1.946e-10,
    ) -> None:
        parameters = {
            "constant": finite_parameter(constant, "constant"),
            "linear": finite_parameter(linear, "linear"),
            "quadratic": finite_parameter(quadratic, "quadratic"),
            "cubic": finite_parameter(cubic, "cubic"),
        }
        arrays = parameter_arrays(parameters)
        self.constant = arrays["constant"]
        self.linear = arrays["linear"]
        self.quadratic = arrays["quadratic"]
        self.cubic = arrays["cubic"]
        self.source_id = "frapcon-4.0-pnnl-19417-rev2:ZOTCON-eq-3.7"
        self.validity = PropertyValidityRange(
            {"temperature": (375.0, 1673.0, "K")}, source_id=self.source_id
        )
        self.law_id = canonical_fingerprint(
            {"kind": "zirconia-thermal-conductivity", "parameters": parameters}
        )

    def thermal_conductivity(self, temperature: ArrayLike, /) -> Array:
        """Oxide conductivity [W/(m K)] at ``temperature`` [K]."""

        t = float64_array(temperature)
        return cubic_polynomial(t, self.constant, self.linear, self.quadratic, self.cubic)


@final
class ZircaloyHydrogenPickup(StrictModule):
    """Hydrogen absorbed by Zircaloy-4 from waterside corrosion in a PWR.

    ``Zr + 2 H2O -> ZrO2 + 2 H2`` releases ``2 M_H / M_O`` kg of hydrogen per
    kg of oxygen taken up; the cladding absorbs the constant pickup fraction
    0.15 of it (FRAPCON-4.0 Section 2.6.2). MATPRO CHUPTK writes the same
    balance with the rounded ratio 1/8 (NUREG/CR-6150 Rev. 2 Vol. 4, Eqs.
    (4-323)-(4-324) and (A2-14)-(A2-16)); standard atomic weights are used here.
    """

    pickup_fraction: Array
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    alloy: CladdingAlloy = eqx.field(static=True, default="zircaloy-4")

    def __init__(self, /, *, pickup_fraction: ArrayLike = 0.15) -> None:
        fraction = finite_parameter(pickup_fraction, "pickup_fraction")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("pickup_fraction must lie in [0, 1].")
        self.pickup_fraction = jnp.asarray(fraction)
        self.source_id = "frapcon-4.0-pnnl-19418-vol1-rev2:section-2.6.2"
        # MATPRO CHUPTK (Section 4.16): typical operation at 523-650 K.
        self.validity = PropertyValidityRange(
            {"temperature": (523.0, 650.0, "K")}, source_id=self.source_id
        )
        self.law_id = canonical_fingerprint(
            {"kind": "zircaloy-4-pwr-hydrogen-pickup", "pickup_fraction": fraction}
        )

    def absorbed_hydrogen_areal_mass(
        self, oxygen_areal_mass_increment: ArrayLike, /
    ) -> Array:
        """Hydrogen absorbed [kg/m^2] per oxygen areal mass taken up [kg/m^2]."""

        return (
            self.pickup_fraction
            * (2.0 * _HYDROGEN_MOLAR_MASS / _OXYGEN_MOLAR_MASS)
            * float64_array(oxygen_areal_mass_increment)
        )

    def hydrogen_content_increment(
        self, oxygen_areal_mass_increment: ArrayLike, metal_areal_mass: ArrayLike, /
    ) -> Array:
        """Hydrogen content increment [ppm by weight] of the cladding metal.

        ``metal_areal_mass`` [kg/m^2] is the metal mass per unit oxidized
        surface area (for a tube oxidized on its outer diameter ``d_o`` with
        inner diameter ``d_i``: ``rho (d_o^2 - d_i^2) / (4 d_o)``).
        """

        return (
            _PARTS_PER_MILLION
            * self.absorbed_hydrogen_areal_mass(oxygen_areal_mass_increment)
            / float64_array(metal_areal_mass)
        )


@final
class ZircaloyEmissivity(StrictModule):
    """Hemispherical emissivity of the oxidized cladding surface (MATPRO ZOEMIS).

    NUREG/CR-6150 Rev. 2 Vol. 4, Eqs. (5-25)-(5-27), (5-31) (identical in
    FRAPCON-4.0, PNNL-19417 Rev. 2 Eqs. (3.9)-(3.11)): ``0.325 + 1.246e5 d``
    for oxide thickness ``d < 3.88e-6 m`` and ``0.808642 - 50 d`` otherwise.
    If the maximum cladding temperature exceeded 1500 K the value is
    multiplied by ``exp((1500 - T_max) / 300)`` and floored at 0.325.
    """

    thin_film_intercept: Array
    thin_film_slope: Array
    thick_film_intercept: Array
    thick_film_slope: Array
    film_transition_thickness: Array
    degradation_temperature: Array
    degradation_scale: Array
    minimum_emissivity: Array
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    alloy: CladdingAlloy = eqx.field(static=True, default="zircaloy-4")

    def __init__(
        self,
        /,
        *,
        thin_film_intercept: ArrayLike = 0.325,
        thin_film_slope: ArrayLike = 1.246e5,
        thick_film_intercept: ArrayLike = 0.808642,
        thick_film_slope: ArrayLike = 50.0,
        film_transition_thickness: ArrayLike = 3.88e-6,
        degradation_temperature: ArrayLike = 1500.0,
        degradation_scale: ArrayLike = 300.0,
        minimum_emissivity: ArrayLike = 0.325,
    ) -> None:
        parameters = {
            "thin_film_intercept": positive_parameter(
                thin_film_intercept, "thin_film_intercept"
            ),
            "thin_film_slope": finite_parameter(thin_film_slope, "thin_film_slope"),
            "thick_film_intercept": positive_parameter(
                thick_film_intercept, "thick_film_intercept"
            ),
            "thick_film_slope": finite_parameter(thick_film_slope, "thick_film_slope"),
            "film_transition_thickness": positive_parameter(
                film_transition_thickness, "film_transition_thickness"
            ),
            "degradation_temperature": positive_parameter(
                degradation_temperature, "degradation_temperature"
            ),
            "degradation_scale": positive_parameter(
                degradation_scale, "degradation_scale"
            ),
            "minimum_emissivity": positive_parameter(
                minimum_emissivity, "minimum_emissivity"
            ),
        }
        arrays = parameter_arrays(parameters)
        self.thin_film_intercept = arrays["thin_film_intercept"]
        self.thin_film_slope = arrays["thin_film_slope"]
        self.thick_film_intercept = arrays["thick_film_intercept"]
        self.thick_film_slope = arrays["thick_film_slope"]
        self.film_transition_thickness = arrays["film_transition_thickness"]
        self.degradation_temperature = arrays["degradation_temperature"]
        self.degradation_scale = arrays["degradation_scale"]
        self.minimum_emissivity = arrays["minimum_emissivity"]
        self.source_id = "matpro-nureg-cr-6150-rev2-vol4:ZOEMIS-eqs-5-25-5-27"
        # MATPRO Section 5.4: data base from Murphy and Havelock (373-673 K),
        # Burgoyne and Garlick, and Juenke and Sjodahl (to 1575 K); oxide
        # layers to 150 um (Figures 5-7 and 5-8).
        self.validity = PropertyValidityRange(
            {
                "oxide_thickness": (0.0, 1.5e-4, "m"),
                "surface_temperature": (373.15, 1575.0, "K"),
            },
            source_id=self.source_id,
        )
        self.law_id = canonical_fingerprint(
            {"kind": "zircaloy-oxide-emissivity", "parameters": parameters}
        )

    def emissivity(
        self, oxide_thickness: ArrayLike, maximum_temperature: ArrayLike, /
    ) -> Array:
        """Emissivity for oxide thickness [m] and maximum cladding temperature [K].

        ``maximum_temperature`` is historical exposure, not instantaneous
        surface temperature. The measurement-temperature interval in
        ``validity`` applies to ``surface_temperature`` supplied by the
        coupled caller, independent of the historical degradation switch.
        For gap radiation the oxide is bore-side oxide; waterside oxide
        thickness is not a bore surface property.
        """

        thickness, peak = (
            float64_array(oxide_thickness),
            float64_array(maximum_temperature),
        )
        film = jnp.where(
            thickness < self.film_transition_thickness,
            self.thin_film_intercept + self.thin_film_slope * thickness,
            self.thick_film_intercept - self.thick_film_slope * thickness,
        )
        degraded = film * jnp.exp(
            (self.degradation_temperature - peak) / self.degradation_scale
        )
        return jnp.where(
            peak > self.degradation_temperature,
            jnp.maximum(degraded, self.minimum_emissivity),
            film,
        )


__all__ = [
    "ZirconiaThermalConductivity",
    "ZircaloyEmissivity",
    "ZircaloyHydrogenPickup",
    "ZircaloyWatersideCorrosion",
]
