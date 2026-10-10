#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Thermal and elastic property laws of stress-relief-annealed Zircaloy-4 cladding.

Every law evaluates its published correlation in the source variables and
converts at the boundary to SI: K, Pa, J, kg, W; fast fluence in n/m^2
(E > 1 MeV). Sources:

- MATPRO: Siefken et al., "SCDAP/RELAP5/MOD 3.3 Code Manual: MATPRO - A Library
  of Materials Properties for Light-Water-Reactor Accident Analysis",
  NUREG/CR-6150 Rev. 2, Vol. 4 (INEL-96/0422), 2001.
- PNNL-19417: Luscher, Geelhood and Porter, "Material Property Correlations:
  Comparisons between FRAPCON-4.0, FRAPTRAN-2.0, and MATPRO", PNNL-19417
  Rev. 2 (NUREG/CR-7024 Rev. 2), 2015, including its Appendix A source listings.
- PNNL-17700: Geelhood, Beyer and Luscher, "PNNL Stress/Strain Correlation for
  Zircaloy", 2008.

Out-of-range evaluation returns the formula value; each ``validity`` states the
fitted domain and the caller decides.
"""

from __future__ import annotations

import math
from typing import final, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...materials import PropertyValidityRange
from ...typing import ConvertibleToArray
from ._identity import CladdingAlloy, CladdingHeatTreatment
from ._material_math import cubic_polynomial
from ._material_parameters import (
    finite_parameter,
    float64_array,
    parameter_arrays,
    positive_parameter,
)


# MATPRO CDEN (Section 4.5.5): Zircaloy density at 300 K reported by Scott,
# WCAP-3269-41 (1965).
ZIRCALOY_DENSITY = 6550.0

# Admissible deviation of the Kearns factor sum from one.
_KEARNS_SUM_TOLERANCE = 1.0e-6

# MATPRO Section 4.1: as-received Zircaloy oxygen weight fraction.
_AS_RECEIVED_OXYGEN = 0.0012

# FRAPCON-4.0 CELMOD/CSHEAR listings (PNNL-19417 Appendix A.2.6): the moduli
# are alpha phase below this temperature [K] whatever the phase boundaries.
ZIRCALOY_ALPHA_PHASE_MAXIMUM_TEMPERATURE = 1090.0
# CELMOD branch switches of the phase boundaries: total oxygen weight fraction
# for the alpha to alpha+beta boundary, excess oxygen for alpha+beta to beta.
_LOWER_BOUNDARY_OXYGEN_SWITCH = 0.025
_UPPER_BOUNDARY_OXYGEN_SWITCH = 4.7308937e-3
# Positivity floor [Pa] of the phase-blended moduli; it keeps the bulk modulus
# and the plastic flow-stress root finite.
_MINIMUM_MODULUS = 1.0


@final
class CladdingPrincipalStrains(StrictModule):
    """Strains along the cladding radial, hoop, and axial directions."""

    radial: Array
    hoop: Array
    axial: Array


@final
class ZircaloyTexture(StrictModule):
    """Kearns texture factors of the cladding tube.

    Each factor is the volume-weighted mean squared cosine between grain
    c-axes and the radial, hoop (tangential), or axial direction; the three sum
    to one. Defaults are the MATPRO typical cladding values ``f_r = 0.66``,
    ``f_theta = 0.28``, ``f_z = 0.06`` (NUREG/CR-6150 Rev. 2 Vol. 4, Section
    4.9.1, from Hann et al., EPRI NP-526 Vol. 3, 1978).
    """

    radial: Array
    hoop: Array
    axial: Array
    source_id: str = eqx.field(static=True)
    texture_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        radial: ArrayLike = 0.66,
        hoop: ArrayLike = 0.28,
        axial: ArrayLike = 0.06,
    ) -> None:
        factors = {
            "radial": finite_parameter(radial, "radial"),
            "hoop": finite_parameter(hoop, "hoop"),
            "axial": finite_parameter(axial, "axial"),
        }
        if any(not 0.0 <= factor <= 1.0 for factor in factors.values()):
            raise ValueError("Kearns texture factors must lie in [0, 1].")
        if abs(math.fsum(factors.values()) - 1.0) > _KEARNS_SUM_TOLERANCE:
            raise ValueError("Kearns texture factors must sum to one.")
        self.radial = jnp.asarray(factors["radial"])
        self.hoop = jnp.asarray(factors["hoop"])
        self.axial = jnp.asarray(factors["axial"])
        self.source_id = "matpro-nureg-cr-6150-rev2-vol4:section-4.9.1"
        self.texture_id = canonical_fingerprint(
            {"kind": "zircaloy-kearns-texture", "factors": factors}
        )


@final
class ZircaloyThermalExpansion(StrictModule):
    """Alpha-phase principal thermal strains of textured cladding (MATPRO CTHEXP).

    Single-crystal strains relative to 300 K (NUREG/CR-6150 Rev. 2 Vol. 4, Eqs.
    (4-27)-(4-28)): perpendicular to the c-axis ``eps_a = 4.95e-6 T -
    1.485e-3`` and parallel to it ``eps_c = 1.26e-5 T - 3.78e-3``. With basal
    symmetry, the pole-figure average of Eqs. (4-34)-(4-36) reduces to
    ``eps_i = eps_a + f_i (eps_c - eps_a)`` for each principal cladding
    direction ``i`` with Kearns factor ``f_i`` of the given ``ZircaloyTexture``.
    Evaluation beyond the alpha range 300-1083 K is
    refused by ``validity``.
    """

    a_axis_slope: Array
    a_axis_intercept: Array
    c_axis_slope: Array
    c_axis_intercept: Array
    texture: ZircaloyTexture
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    alloy: CladdingAlloy = eqx.field(static=True, default="zircaloy-4")
    heat_treatment: CladdingHeatTreatment = eqx.field(
        static=True, default="stress-relief-annealed"
    )

    def __init__(
        self,
        /,
        *,
        texture: ZircaloyTexture | None = None,
        a_axis_slope: ArrayLike = 4.95e-6,
        a_axis_intercept: ArrayLike = -1.485e-3,
        c_axis_slope: ArrayLike = 1.26e-5,
        c_axis_intercept: ArrayLike = -3.78e-3,
    ) -> None:
        texture_law = ZircaloyTexture() if texture is None else texture
        if not isinstance(texture_law, ZircaloyTexture):
            raise TypeError("texture must be a ZircaloyTexture.")
        parameters = {
            "a_axis_slope": positive_parameter(a_axis_slope, "a_axis_slope"),
            "a_axis_intercept": finite_parameter(a_axis_intercept, "a_axis_intercept"),
            "c_axis_slope": positive_parameter(c_axis_slope, "c_axis_slope"),
            "c_axis_intercept": finite_parameter(c_axis_intercept, "c_axis_intercept"),
        }
        arrays = parameter_arrays(parameters)
        self.a_axis_slope = arrays["a_axis_slope"]
        self.a_axis_intercept = arrays["a_axis_intercept"]
        self.c_axis_slope = arrays["c_axis_slope"]
        self.c_axis_intercept = arrays["c_axis_intercept"]
        self.texture = texture_law
        self.source_id = "matpro-nureg-cr-6150-rev2-vol4:CTHEXP-eqs-4-27-4-28-4-34-4-36"
        self.validity = PropertyValidityRange(
            {"temperature": (300.0, 1083.0, "K")},
            source_id=self.source_id,
            exceedance="refuse",
        )
        self.law_id = canonical_fingerprint(
            {
                "kind": "zircaloy-thermal-expansion",
                "parameters": parameters,
                "texture": texture_law.texture_id,
            }
        )

    def principal_strains(self, temperature: ArrayLike, /) -> CladdingPrincipalStrains:
        """Thermal strains [m/m] relative to 300 K at ``temperature`` [K]."""

        t = float64_array(temperature)
        basal = self.a_axis_slope * t + self.a_axis_intercept
        anisotropy = self.c_axis_slope * t + self.c_axis_intercept - basal
        return CladdingPrincipalStrains(
            radial=basal + self.texture.radial * anisotropy,
            hoop=basal + self.texture.hoop * anisotropy,
            axial=basal + self.texture.axial * anisotropy,
        )


@final
class ZircaloyThermalConductivity(StrictModule):
    """Zircaloy thermal conductivity [W/(m K)] (CTHCON).

    ``k = 7.511 + 2.088e-2 T - 1.450e-5 T^2 + 7.668e-9 T^3`` as coded in the
    FRAPCON-4.0 CTHCON listing (PNNL-19417 Rev. 2 Appendix A.2.2); MATPRO Eq.
    (4-26) prints the same fit rounded to three digits. Data extend from room
    temperature to about 1800 K (PNNL-19417 Section 3.2.1).
    """

    constant: Array
    linear: Array
    quadratic: Array
    cubic: Array
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    alloy: CladdingAlloy = eqx.field(static=True, default="zircaloy-4")

    def __init__(
        self,
        /,
        *,
        constant: ArrayLike = 7.511,
        linear: ArrayLike = 2.088e-2,
        quadratic: ArrayLike = -1.450e-5,
        cubic: ArrayLike = 7.668e-9,
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
        self.source_id = "frapcon-4.0-pnnl-19417-rev2:CTHCON"
        self.validity = PropertyValidityRange(
            {"temperature": (300.0, 1800.0, "K")}, source_id=self.source_id
        )
        self.law_id = canonical_fingerprint(
            {"kind": "zircaloy-thermal-conductivity", "parameters": parameters}
        )

    def thermal_conductivity(self, temperature: ArrayLike, /) -> Array:
        """Conductivity [W/(m K)] at ``temperature`` [K]."""

        t = float64_array(temperature)
        return cubic_polynomial(t, self.constant, self.linear, self.quadratic, self.cubic)


# FRAPCON-4.0 / MATPRO CCP interpolation table (PNNL-19417 Table 3.2; MATPRO
# Table 4-2): temperatures [K] and specific heat capacities [J/(kg K)].
_HEAT_CAPACITY_TEMPERATURES = (
    300.0,
    400.0,
    640.0,
    1090.0,
    1093.0,
    1113.0,
    1133.0,
    1153.0,
    1173.0,
    1193.0,
    1213.0,
    1233.0,
    1248.0,
)
_HEAT_CAPACITY_VALUES = (
    281.0,
    302.0,
    331.0,
    375.0,
    502.0,
    590.0,
    615.0,
    719.0,
    816.0,
    770.0,
    619.0,
    469.0,
    356.0,
)


class _HeatCapacitySegment(NamedTuple):
    """Table interval containing each temperature, with its knots and slope."""

    temperature: Array
    knots: Array
    values: Array
    index: Array
    start: Array
    slope: Array


@final
class ZircaloyHeatCapacity(StrictModule):
    """Zircaloy specific heat capacity and enthalpy (CCP, CENTHL).

    ``cp`` is the linear interpolant of PNNL-19417 Table 3.2 (Brooks and
    Stansbury below 1090 K, Deem and Eldridge through the alpha-beta
    transformation) and is constant at the last value above 1248 K, as in the
    FRAPCON-4.0 CCP listing. The enthalpy relative to 300 K is its exact
    integral, MATPRO CENTHL Eq. (4-21), so ``d h / d T = cp``.
    """

    specific_heat_capacities: Array
    validity: PropertyValidityRange
    temperatures: tuple[float, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    alloy: CladdingAlloy = eqx.field(static=True, default="zircaloy-4")

    def __init__(
        self,
        /,
        *,
        specific_heat_capacities: ConvertibleToArray = _HEAT_CAPACITY_VALUES,
    ) -> None:
        values = np.asarray(specific_heat_capacities, dtype=np.float64)
        if values.shape != (len(_HEAT_CAPACITY_TEMPERATURES),):
            raise ValueError(
                "specific_heat_capacities must list one value per CCP table "
                f"temperature ({len(_HEAT_CAPACITY_TEMPERATURES)})."
            )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Specific heat capacities must be finite and positive.")
        self.specific_heat_capacities = jnp.asarray(values)
        self.temperatures = _HEAT_CAPACITY_TEMPERATURES
        self.source_id = "frapcon-4.0-pnnl-19417-rev2:CCP-table-3.2"
        self.validity = PropertyValidityRange(
            {"temperature": (300.0, 2098.0, "K")}, source_id=self.source_id
        )
        self.law_id = canonical_fingerprint(
            {
                "kind": "zircaloy-heat-capacity",
                "temperatures": list(_HEAT_CAPACITY_TEMPERATURES),
                "specific_heat_capacities": values,
            }
        )

    def _segment(self, temperature: ArrayLike, /) -> _HeatCapacitySegment:
        t = float64_array(temperature)
        knots = jnp.asarray(self.temperatures)
        values = self.specific_heat_capacities
        index = jnp.clip(jnp.searchsorted(knots, t, side="right") - 1, 0, knots.size - 2)
        start = knots[index]
        slope = (values[index + 1] - values[index]) / (knots[index + 1] - start)
        return _HeatCapacitySegment(
            temperature=t,
            knots=knots,
            values=values,
            index=index,
            start=start,
            slope=slope,
        )

    def specific_heat_capacity(self, temperature: ArrayLike, /) -> Array:
        """Specific heat capacity [J/(kg K)] at ``temperature`` [K]."""

        t, knots, values, index, start, slope = self._segment(temperature)
        interior = values[index] + slope * (t - start)
        return jnp.where(
            t >= knots[-1], values[-1], jnp.where(t < knots[0], values[0], interior)
        )

    def specific_enthalpy(self, temperature: ArrayLike, /) -> Array:
        """Specific enthalpy [J/kg] relative to 300 K at ``temperature`` [K]."""

        t, knots, values, index, start, slope = self._segment(temperature)
        segment_enthalpy = 0.5 * (values[1:] + values[:-1]) * jnp.diff(knots)
        cumulative = jnp.concatenate(
            [jnp.zeros((1,), dtype=values.dtype), jnp.cumsum(segment_enthalpy)]
        )
        offset = t - start
        interior = cumulative[index] + values[index] * offset + 0.5 * slope * offset**2
        above = cumulative[-1] + values[-1] * (t - knots[-1])
        below = values[0] * (t - knots[0])
        return jnp.where(t >= knots[-1], above, jnp.where(t < knots[0], below, interior))


def _phase_blended_modulus(
    temperature: Array,
    alpha: Array,
    alpha_at_lower_boundary: Array,
    beta: Array,
    beta_at_upper_boundary: Array,
    lower_boundary: Array,
    upper_boundary: Array,
) -> Array:
    # FRAPCON-4.0 CELMOD/CSHEAR listings: alpha below 1090 K or the alpha to
    # alpha+beta boundary, beta above the alpha+beta to beta boundary, linear
    # between the boundary values. The result is floored at _MINIMUM_MODULUS.
    blended = alpha_at_lower_boundary + (temperature - lower_boundary) * (
        beta_at_upper_boundary - alpha_at_lower_boundary
    ) / (upper_boundary - lower_boundary)
    modulus = jnp.where(
        (temperature < ZIRCALOY_ALPHA_PHASE_MAXIMUM_TEMPERATURE)
        | (temperature < lower_boundary),
        alpha,
        jnp.where(temperature > upper_boundary, beta, blended),
    )
    return jnp.maximum(modulus, _MINIMUM_MODULUS)


def _phase_boundaries(excess_oxygen: Array, /) -> tuple[Array, Array]:
    # FRAPCON-4.0 CELMOD listing (PNNL-19417 Appendix A.2.6): alpha to
    # alpha+beta (taab) and alpha+beta to beta (tabb) temperatures [K].
    total_oxygen = excess_oxygen + _AS_RECEIVED_OXYGEN
    lower = jnp.where(
        total_oxygen < _LOWER_BOUNDARY_OXYGEN_SWITCH,
        1094.0 + total_oxygen * (-1.289e3 + total_oxygen * 7.914e5),
        1556.4 + 3.8281e4 * (total_oxygen - _LOWER_BOUNDARY_OXYGEN_SWITCH),
    )
    upper = jnp.where(
        excess_oxygen < _UPPER_BOUNDARY_OXYGEN_SWITCH,
        392.46 * ((100.0 * excess_oxygen + 0.1242807) ** 2 + 3.1417),
        (100.0 * excess_oxygen + 0.12) * 491.157 + 1081.7413,
    )
    return lower, upper


@final
class ZircaloyElasticModuli(StrictModule):
    """Young's, shear, and bulk moduli [Pa] of isotropic Zircaloy."""

    youngs: Array
    shear: Array
    bulk: Array


@final
class ZircaloyElasticity(StrictModule):
    """Isotropic Young's and shear moduli of Zircaloy (MATPRO CELMOD, CSHEAR).

    Alpha phase (NUREG/CR-6150 Rev. 2 Vol. 4, Eqs. (4-71), (4-73)-(4-76),
    (4-78)):

    ``E = (1.088e11 - 5.475e7 T + (6.61e11 + 5.912e8 T) D - 2.6e10 C) / K3``
    ``G = (4.04e10 - 2.168e7 T + (7.07e11 - 2.315e8 T) D - 8.67e9 C) / K3``
    ``K3 = 0.88 + 0.12 exp(-Phi / 1e25)``

    with ``D`` the average oxygen mass fraction above the as-received 0.0012,
    ``C`` the cold work (area-reduction ratio) and ``Phi`` the fast fluence
    [n/m^2]. Beta phase ``E = 9.21e10 - 4.05e7 T`` and ``G = 3.49e10 -
    1.66e7 T`` (Eqs. (4-72), (4-77)), linear across the alpha+beta range
    between the phase boundaries of the FRAPCON-4.0 listings. Poisson's ratio
    is ``E / (2 G) - 1``.

    The selected alpha, beta, or blended modulus is floored at 1 Pa with
    ``jnp.maximum``; where the floor binds the modulus is 1 Pa with zero
    derivative, and ``validity`` does not report it. At the default
    coefficients the floor binds nowhere inside ``validity`` at zero excess
    oxygen: the smallest modulus there is the beta shear modulus, about 7e7 Pa
    at 2098 K. It can bind for calibrated coefficients, and for excess oxygen
    above about 0.0196, where the alpha+beta to beta boundary exceeds 2102 K,
    the beta shear formula is negative at that boundary, and the blended
    shear modulus therefore reaches zero below it.
    """

    youngs_intercept: Array
    youngs_slope: Array
    youngs_oxygen_intercept: Array
    youngs_oxygen_slope: Array
    youngs_cold_work: Array
    youngs_beta_intercept: Array
    youngs_beta_slope: Array
    shear_intercept: Array
    shear_slope: Array
    shear_oxygen_intercept: Array
    shear_oxygen_slope: Array
    shear_cold_work: Array
    shear_beta_intercept: Array
    shear_beta_slope: Array
    irradiated_modulus_ratio: Array
    fluence_scale: Array
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    alloy: CladdingAlloy = eqx.field(static=True, default="zircaloy-4")

    def __init__(
        self,
        /,
        *,
        youngs_intercept: ArrayLike = 1.088e11,
        youngs_slope: ArrayLike = 5.475e7,
        youngs_oxygen_intercept: ArrayLike = 6.61e11,
        youngs_oxygen_slope: ArrayLike = 5.912e8,
        youngs_cold_work: ArrayLike = 2.6e10,
        youngs_beta_intercept: ArrayLike = 9.21e10,
        youngs_beta_slope: ArrayLike = 4.05e7,
        shear_intercept: ArrayLike = 4.04e10,
        shear_slope: ArrayLike = 2.168e7,
        shear_oxygen_intercept: ArrayLike = 7.07e11,
        shear_oxygen_slope: ArrayLike = -2.315e8,
        shear_cold_work: ArrayLike = 8.67e9,
        shear_beta_intercept: ArrayLike = 3.49e10,
        shear_beta_slope: ArrayLike = 1.66e7,
        irradiated_modulus_ratio: ArrayLike = 0.88,
        fluence_scale: ArrayLike = 1.0e25,
    ) -> None:
        parameters = {
            "youngs_intercept": positive_parameter(youngs_intercept, "youngs_intercept"),
            "youngs_slope": finite_parameter(youngs_slope, "youngs_slope"),
            "youngs_oxygen_intercept": finite_parameter(
                youngs_oxygen_intercept, "youngs_oxygen_intercept"
            ),
            "youngs_oxygen_slope": finite_parameter(
                youngs_oxygen_slope, "youngs_oxygen_slope"
            ),
            "youngs_cold_work": finite_parameter(youngs_cold_work, "youngs_cold_work"),
            "youngs_beta_intercept": positive_parameter(
                youngs_beta_intercept, "youngs_beta_intercept"
            ),
            "youngs_beta_slope": finite_parameter(youngs_beta_slope, "youngs_beta_slope"),
            "shear_intercept": positive_parameter(shear_intercept, "shear_intercept"),
            "shear_slope": finite_parameter(shear_slope, "shear_slope"),
            "shear_oxygen_intercept": finite_parameter(
                shear_oxygen_intercept, "shear_oxygen_intercept"
            ),
            "shear_oxygen_slope": finite_parameter(
                shear_oxygen_slope, "shear_oxygen_slope"
            ),
            "shear_cold_work": finite_parameter(shear_cold_work, "shear_cold_work"),
            "shear_beta_intercept": positive_parameter(
                shear_beta_intercept, "shear_beta_intercept"
            ),
            "shear_beta_slope": finite_parameter(shear_beta_slope, "shear_beta_slope"),
            "irradiated_modulus_ratio": positive_parameter(
                irradiated_modulus_ratio, "irradiated_modulus_ratio"
            ),
            "fluence_scale": positive_parameter(fluence_scale, "fluence_scale"),
        }
        arrays = parameter_arrays(parameters)
        self.youngs_intercept = arrays["youngs_intercept"]
        self.youngs_slope = arrays["youngs_slope"]
        self.youngs_oxygen_intercept = arrays["youngs_oxygen_intercept"]
        self.youngs_oxygen_slope = arrays["youngs_oxygen_slope"]
        self.youngs_cold_work = arrays["youngs_cold_work"]
        self.youngs_beta_intercept = arrays["youngs_beta_intercept"]
        self.youngs_beta_slope = arrays["youngs_beta_slope"]
        self.shear_intercept = arrays["shear_intercept"]
        self.shear_slope = arrays["shear_slope"]
        self.shear_oxygen_intercept = arrays["shear_oxygen_intercept"]
        self.shear_oxygen_slope = arrays["shear_oxygen_slope"]
        self.shear_cold_work = arrays["shear_cold_work"]
        self.shear_beta_intercept = arrays["shear_beta_intercept"]
        self.shear_beta_slope = arrays["shear_beta_slope"]
        self.irradiated_modulus_ratio = arrays["irradiated_modulus_ratio"]
        self.fluence_scale = arrays["fluence_scale"]
        self.source_id = "matpro-nureg-cr-6150-rev2-vol4:CELMOD-CSHEAR-eqs-4-71-4-78"
        # Moduli are defined below the 2098 K melting point (PNNL-19417 Section
        # 3.6.2); cold work 0-0.75 (PNNL-17700 Section 2.1).
        self.validity = PropertyValidityRange(
            {"temperature": (300.0, 2098.0, "K"), "cold_work": (0.0, 0.75, "1")},
            source_id=self.source_id,
        )
        self.law_id = canonical_fingerprint(
            {"kind": "zircaloy-elasticity", "parameters": parameters}
        )

    def _fluence_factor(self, fast_fluence: Array, /) -> Array:
        return self.irradiated_modulus_ratio + (
            1.0 - self.irradiated_modulus_ratio
        ) * jnp.exp(-fast_fluence / self.fluence_scale)

    def _phase_modulus(
        self,
        temperature: ArrayLike,
        fast_fluence: ArrayLike,
        cold_work: ArrayLike,
        excess_oxygen: ArrayLike,
        /,
        *,
        intercept: Array,
        slope: Array,
        oxygen_intercept: Array,
        oxygen_slope: Array,
        cold_work_coefficient: Array,
        beta_intercept: Array,
        beta_slope: Array,
    ) -> Array:
        """Alpha, beta, and alpha+beta blended modulus [Pa] for one coefficient set."""

        t, fluence, work, oxygen = (
            float64_array(temperature),
            float64_array(fast_fluence),
            float64_array(cold_work),
            float64_array(excess_oxygen),
        )
        factor = self._fluence_factor(fluence)

        def alpha_modulus(at_temperature: Array, /) -> Array:
            return (
                intercept
                - slope * at_temperature
                + (oxygen_intercept + oxygen_slope * at_temperature) * oxygen
                - cold_work_coefficient * work
            ) / factor

        def beta_modulus(at_temperature: Array, /) -> Array:
            return beta_intercept - beta_slope * at_temperature

        lower, upper = _phase_boundaries(oxygen)
        return _phase_blended_modulus(
            t,
            alpha_modulus(t),
            alpha_modulus(lower),
            beta_modulus(t),
            beta_modulus(upper),
            lower,
            upper,
        )

    def youngs_modulus(
        self,
        temperature: ArrayLike,
        fast_fluence: ArrayLike,
        cold_work: ArrayLike,
        excess_oxygen: ArrayLike,
        /,
    ) -> Array:
        """Young's modulus [Pa] of randomly textured Zircaloy."""

        return self._phase_modulus(
            temperature,
            fast_fluence,
            cold_work,
            excess_oxygen,
            intercept=self.youngs_intercept,
            slope=self.youngs_slope,
            oxygen_intercept=self.youngs_oxygen_intercept,
            oxygen_slope=self.youngs_oxygen_slope,
            cold_work_coefficient=self.youngs_cold_work,
            beta_intercept=self.youngs_beta_intercept,
            beta_slope=self.youngs_beta_slope,
        )

    def shear_modulus(
        self,
        temperature: ArrayLike,
        fast_fluence: ArrayLike,
        cold_work: ArrayLike,
        excess_oxygen: ArrayLike,
        /,
    ) -> Array:
        """Shear modulus [Pa] of randomly textured Zircaloy."""

        return self._phase_modulus(
            temperature,
            fast_fluence,
            cold_work,
            excess_oxygen,
            intercept=self.shear_intercept,
            slope=self.shear_slope,
            oxygen_intercept=self.shear_oxygen_intercept,
            oxygen_slope=self.shear_oxygen_slope,
            cold_work_coefficient=self.shear_cold_work,
            beta_intercept=self.shear_beta_intercept,
            beta_slope=self.shear_beta_slope,
        )

    def poisson_ratio(
        self,
        temperature: ArrayLike,
        fast_fluence: ArrayLike,
        cold_work: ArrayLike,
        excess_oxygen: ArrayLike,
        /,
    ) -> Array:
        """Isotropic Poisson's ratio ``E / (2 G) - 1``."""

        youngs = self.youngs_modulus(temperature, fast_fluence, cold_work, excess_oxygen)
        shear = self.shear_modulus(temperature, fast_fluence, cold_work, excess_oxygen)
        return youngs / (2.0 * shear) - 1.0

    def moduli(
        self,
        temperature: ArrayLike,
        fast_fluence: ArrayLike,
        cold_work: ArrayLike,
        excess_oxygen: ArrayLike,
        /,
    ) -> ZircaloyElasticModuli:
        """Young's, shear, and bulk ``E / (3 (1 - 2 nu))`` moduli [Pa]."""

        shear = self.shear_modulus(temperature, fast_fluence, cold_work, excess_oxygen)
        youngs = self.youngs_modulus(temperature, fast_fluence, cold_work, excess_oxygen)
        poisson = youngs / (2.0 * shear) - 1.0
        return ZircaloyElasticModuli(
            youngs=youngs, shear=shear, bulk=youngs / (3.0 * (1.0 - 2.0 * poisson))
        )


__all__ = [
    "ZIRCALOY_ALPHA_PHASE_MAXIMUM_TEMPERATURE",
    "ZIRCALOY_DENSITY",
    "CladdingPrincipalStrains",
    "ZircaloyElasticity",
    "ZircaloyElasticModuli",
    "ZircaloyHeatCapacity",
    "ZircaloyTexture",
    "ZircaloyThermalConductivity",
    "ZircaloyThermalExpansion",
]
