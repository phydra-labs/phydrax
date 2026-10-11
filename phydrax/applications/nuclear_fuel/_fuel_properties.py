#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Thermophysical and elastic property laws of sintered UO2 fuel pellets.

Every law evaluates its published correlation in the correlation's own
variables; unit conversions happen inside the law. Burnup enters as
``burnup_mwd_per_kghm``: the heavy metal of UO2 fuel is uranium, so one
MWd/kgHM is one MWd/kgU, which is one GWd/MTU (the burnup unit of the FRAPCON
correlations). Out-of-range inputs still return the correlation value; each
law's ``validity`` states the source envelope and the caller decides.

Primary sources:

- Fink, "Thermophysical properties of uranium dioxide", J. Nucl. Mater. 279
  (2000) 1-18, recommended equations as reproduced in IAEA-TECDOC-1496 (2006),
  Section 6.1.1.1, Eqs. (1)-(2) and Tables 1 and 3.
- Luscher, Geelhood, Porter, "Material Property Correlations: Comparisons
  between FRAPCON-4.0, FRAPTRAN-2.0, and MATPRO", PNNL-19417 Rev. 2 (2015).
- Geelhood et al., "FRAPCON-4.0: A Computer Code for the Calculation of
  Steady-State, Thermal-Mechanical Behavior of Oxide Fuel Rods for High
  Burnup", PNNL-19418 Vol. 1 Rev. 2 (2015).
- Siefken et al., "MATPRO - A Library of Materials Properties for
  Light-Water-Reactor Accident Analysis", NUREG/CR-6150 Vol. 4 (SCDAP/RELAP5
  code manual), Sections 2.6 (FELMOD, FPOIR).
- Geelhood et al., "MatLib-1.2.1: Nuclear Material Properties Library",
  PNNL-35702 (2024), for the stated applicability ranges.
"""

from __future__ import annotations

from typing import Final, final

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...materials import PropertyValidityRange
from ._material_math import zero_safe_log_power
from ._material_parameters import finite_parameter, float64_array, positive_parameter


# FRAPCON-4.0 (PNNL-19418 Vol. 1 Rev. 2, Appendix A, input ``den``): "Theoretical
# density taken as 10.96 g/cm3"; MatLib-1.2.1 (PNNL-35702) Section 2.1.6.1:
# "The theoretical density of pure UO2 is taken as 10960 kg/m3".
UO2_THEORETICAL_DENSITY: Final[float] = 10960.0

# Fink (2000) enthalpy increments are referenced to 298.15 K.
_ENTHALPY_REFERENCE_TEMPERATURE = 298.15
# Boltzmann constant exactly as used by FRAPCON-4.0 FTHEXP (PNNL-19417 Rev. 2,
# Eq. (2.24) and Appendix A.1.5 ``bk = 1.38e-23``); the fitted defect energy is
# paired with this value, so the source value is kept rather than CODATA.
_FTHEXP_BOLTZMANN_CONSTANT = 1.38e-23


@final
class UO2HeatCapacity(StrictModule):
    """Enthalpy and isobaric heat capacity of solid UO2 (Fink 2000).

    Fink's recommended enthalpy increment (IAEA-TECDOC-1496, Section 6.1.1.1,
    Eq. (1)), per mole of UO2 and with ``x = theta / T``::

        H(T) - H(298.15 K) = C1 theta [1/(e^x - 1) - 1/(e^(theta/298.15) - 1)]
                             + C2 (T^2 - 298.15^2) + C3 exp(-Ea/T)

    and its exact temperature derivative (Eq. (2))::

        cp(T) = C1 x^2 e^x / (e^x - 1)^2 + 2 C2 T + C3 Ea exp(-Ea/T) / T^2

    so ``heat_capacity`` is exactly the derivative of ``enthalpy``. The
    defect term is also referenced to 298.15 K, making the enthalpy exactly
    zero there for all calibrations (the omitted reference term in the
    publication is below 1e-19 J/mol at the source coefficients). Per-mass
    values divide by ``molar_mass``; the default uses IUPAC standard atomic
    weights (U 238.02891, O 15.9994 g/mol), and is calibratable because
    enrichment shifts it slightly.
    """

    einstein_coefficient: Array
    einstein_temperature: Array
    lattice_coefficient: Array
    defect_coefficient: Array
    defect_temperature: Array
    molar_mass: Array
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        einstein_coefficient: ArrayLike = 81.613,
        einstein_temperature: ArrayLike = 548.68,
        lattice_coefficient: ArrayLike = 2.285e-3,
        defect_coefficient: ArrayLike = 2.360e7,
        defect_temperature: ArrayLike = 18531.7,
        molar_mass: ArrayLike = 0.27002771,
    ) -> None:
        parameters = {
            "einstein_coefficient": positive_parameter(
                einstein_coefficient, "einstein_coefficient"
            ),
            "einstein_temperature": positive_parameter(
                einstein_temperature, "einstein_temperature"
            ),
            "lattice_coefficient": finite_parameter(
                lattice_coefficient, "lattice_coefficient"
            ),
            "defect_coefficient": finite_parameter(
                defect_coefficient, "defect_coefficient"
            ),
            "defect_temperature": positive_parameter(
                defect_temperature, "defect_temperature"
            ),
            "molar_mass": positive_parameter(molar_mass, "molar_mass"),
        }
        self.einstein_coefficient = jnp.asarray(parameters["einstein_coefficient"])
        self.einstein_temperature = jnp.asarray(parameters["einstein_temperature"])
        self.lattice_coefficient = jnp.asarray(parameters["lattice_coefficient"])
        self.defect_coefficient = jnp.asarray(parameters["defect_coefficient"])
        self.defect_temperature = jnp.asarray(parameters["defect_temperature"])
        self.molar_mass = jnp.asarray(parameters["molar_mass"])
        # IAEA-TECDOC-1496 Section 6.1.1.1: "for 298.15 K < T < 3120 K".
        self.validity = PropertyValidityRange(
            {"temperature": (_ENTHALPY_REFERENCE_TEMPERATURE, 3120.0, "K")},
            source_id="iaea-tecdoc-1496:6.1.1.1",
        )
        self.source_id = "fink-2000-jnm-279-1:iaea-tecdoc-1496-6.1.1.1-eq-1-2"
        self.law_id = canonical_fingerprint(
            {
                "kind": "uo2-heat-capacity-fink-2000",
                "parameters": parameters,
                "source": self.source_id,
            }
        )

    def enthalpy(self, temperature: ArrayLike, /) -> Array:
        """Specific enthalpy ``h(T) - h(298.15 K)`` [J/kg] at temperature [K]."""

        t = float64_array(temperature)
        theta = self.einstein_temperature
        einstein = theta * (
            1.0 / jnp.expm1(theta / t)
            - 1.0 / jnp.expm1(theta / _ENTHALPY_REFERENCE_TEMPERATURE)
        )
        molar = (
            self.einstein_coefficient * einstein
            + self.lattice_coefficient
            * (t * t - _ENTHALPY_REFERENCE_TEMPERATURE * _ENTHALPY_REFERENCE_TEMPERATURE)
            + self.defect_coefficient
            * (
                jnp.exp(-self.defect_temperature / t)
                - jnp.exp(-self.defect_temperature / _ENTHALPY_REFERENCE_TEMPERATURE)
            )
        )
        return molar / self.molar_mass

    def heat_capacity(self, temperature: ArrayLike, /) -> Array:
        """Specific isobaric heat capacity ``dh/dT`` [J/(kg K)] at temperature [K]."""

        t = float64_array(temperature)
        x = self.einstein_temperature / t
        # x^2 e^x / (e^x - 1)^2 written as (x / (2 sinh(x/2)))^2: overflow-free.
        einstein = jnp.square(x / (2.0 * jnp.sinh(0.5 * x)))
        molar = (
            self.einstein_coefficient * einstein
            + 2.0 * self.lattice_coefficient * t
            + self.defect_coefficient
            * self.defect_temperature
            * jnp.exp(-self.defect_temperature / t)
            / (t * t)
        )
        return molar / self.molar_mass


@final
class UO2ThermalConductivity(StrictModule):
    """Modified NFI thermal conductivity of UO2 and UO2-Gd2O3 (FRAPCON-4.0).

    PNNL-19417 Rev. 2, Eq. (2.19) (FRAPCON-4.0 Eq. (2.52)) for 95% dense fuel::

        K95 = 1 / (A + a gad + B T + f(Bu) + (1 - 0.9 exp(-0.04 Bu)) g(Bu) h(T))
              + E / T^2 exp(-F/T)

    with ``f = 0.00187 Bu``, ``g = 0.038 Bu^0.28``,
    ``h = 1 / (1 + 396 exp(-Q/T))`` and ``Bu`` in GWd/MTU, adjusted to the
    fuel density by the Lucuta spherical-pore factor of Eq. (2.20)::

        K = 1.0789 K95 d / (1 + 0.5 (1 - d)),   d = 1 - porosity

    ``1.0789 = 1.025 / 0.95`` makes the factor one at 95% of theoretical
    density. FRAPCON-4.0 applies it to the as-fabricated density; MatLib-1.2.1
    Eq. (2-4) includes densification in ``d``. The caller supplies only these
    pores, not gas-bubble or HBS porosity. The source validity's
    ``as_fabricated_porosity`` applies to fabrication, not the evolving ``d``.
    ``Bu^0.28`` has an unbounded burnup derivative at zero
    burnup; the law evaluates that term as exactly zero at ``Bu = 0`` with a
    zero burnup derivative there so parameter gradients stay finite.
    """

    phonon_constant: Array
    gadolinia_coefficient: Array
    phonon_temperature_coefficient: Array
    fission_product_coefficient: Array
    defect_coefficient: Array
    defect_burnup_exponent: Array
    defect_recovery_fraction: Array
    defect_recovery_rate: Array
    annealing_prefactor: Array
    annealing_temperature: Array
    electronic_coefficient: Array
    electronic_temperature: Array
    porosity_normalization: Array
    pore_shape_coefficient: Array
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        phonon_constant: ArrayLike = 0.0452,
        gadolinia_coefficient: ArrayLike = 1.1599,
        phonon_temperature_coefficient: ArrayLike = 2.46e-4,
        fission_product_coefficient: ArrayLike = 0.00187,
        defect_coefficient: ArrayLike = 0.038,
        defect_burnup_exponent: ArrayLike = 0.28,
        defect_recovery_fraction: ArrayLike = 0.9,
        defect_recovery_rate: ArrayLike = 0.04,
        annealing_prefactor: ArrayLike = 396.0,
        annealing_temperature: ArrayLike = 6380.0,
        electronic_coefficient: ArrayLike = 3.5e9,
        electronic_temperature: ArrayLike = 16361.0,
        porosity_normalization: ArrayLike = 1.0789,
        pore_shape_coefficient: ArrayLike = 0.5,
    ) -> None:
        parameters = {
            "phonon_constant": positive_parameter(phonon_constant, "phonon_constant"),
            "gadolinia_coefficient": finite_parameter(
                gadolinia_coefficient, "gadolinia_coefficient"
            ),
            "phonon_temperature_coefficient": finite_parameter(
                phonon_temperature_coefficient, "phonon_temperature_coefficient"
            ),
            "fission_product_coefficient": finite_parameter(
                fission_product_coefficient, "fission_product_coefficient"
            ),
            "defect_coefficient": finite_parameter(
                defect_coefficient, "defect_coefficient"
            ),
            "defect_burnup_exponent": positive_parameter(
                defect_burnup_exponent, "defect_burnup_exponent"
            ),
            "defect_recovery_fraction": finite_parameter(
                defect_recovery_fraction, "defect_recovery_fraction"
            ),
            "defect_recovery_rate": finite_parameter(
                defect_recovery_rate, "defect_recovery_rate"
            ),
            "annealing_prefactor": finite_parameter(
                annealing_prefactor, "annealing_prefactor"
            ),
            "annealing_temperature": finite_parameter(
                annealing_temperature, "annealing_temperature"
            ),
            "electronic_coefficient": finite_parameter(
                electronic_coefficient, "electronic_coefficient"
            ),
            "electronic_temperature": finite_parameter(
                electronic_temperature, "electronic_temperature"
            ),
            "porosity_normalization": positive_parameter(
                porosity_normalization, "porosity_normalization"
            ),
            "pore_shape_coefficient": finite_parameter(
                pore_shape_coefficient, "pore_shape_coefficient"
            ),
        }
        self.phonon_constant = jnp.asarray(parameters["phonon_constant"])
        self.gadolinia_coefficient = jnp.asarray(parameters["gadolinia_coefficient"])
        self.phonon_temperature_coefficient = jnp.asarray(
            parameters["phonon_temperature_coefficient"]
        )
        self.fission_product_coefficient = jnp.asarray(
            parameters["fission_product_coefficient"]
        )
        self.defect_coefficient = jnp.asarray(parameters["defect_coefficient"])
        self.defect_burnup_exponent = jnp.asarray(parameters["defect_burnup_exponent"])
        self.defect_recovery_fraction = jnp.asarray(
            parameters["defect_recovery_fraction"]
        )
        self.defect_recovery_rate = jnp.asarray(parameters["defect_recovery_rate"])
        self.annealing_prefactor = jnp.asarray(parameters["annealing_prefactor"])
        self.annealing_temperature = jnp.asarray(parameters["annealing_temperature"])
        self.electronic_coefficient = jnp.asarray(parameters["electronic_coefficient"])
        self.electronic_temperature = jnp.asarray(parameters["electronic_temperature"])
        self.porosity_normalization = jnp.asarray(parameters["porosity_normalization"])
        self.pore_shape_coefficient = jnp.asarray(parameters["pore_shape_coefficient"])
        # PNNL-19417 Rev. 2 Section 2.3.2: 300-3000 K, rod-average burnup
        # 0-62 GWd/MTU, as-fabricated density 92-97 %TD, gadolinia 0-10 wt%.
        self.validity = PropertyValidityRange(
            {
                "temperature": (300.0, 3000.0, "K"),
                "burnup_mwd_per_kghm": (0.0, 62.0, "MWd/kgHM"),
                "as_fabricated_porosity": (0.03, 0.08, "1"),
                "gadolinia_weight_fraction": (0.0, 0.10, "1"),
            },
            source_id="pnnl-19417-rev2:2.3.2",
        )
        self.source_id = "pnnl-19417-rev2:eq-2.19-2.20"
        self.law_id = canonical_fingerprint(
            {
                "kind": "uo2-thermal-conductivity-modified-nfi",
                "parameters": parameters,
                "source": self.source_id,
            }
        )

    def conductivity(
        self,
        temperature: ArrayLike,
        burnup_mwd_per_kghm: ArrayLike,
        porosity: ArrayLike,
        gadolinia_weight_fraction: ArrayLike,
        /,
    ) -> Array:
        """Thermal conductivity [W/(m K)] of fuel with the given porosity fraction."""

        t, burnup, pores, gadolinia = jnp.broadcast_arrays(
            float64_array(temperature),
            float64_array(burnup_mwd_per_kghm),
            float64_array(porosity),
            float64_array(gadolinia_weight_fraction),
        )
        # MWd/kgHM equals GWd/MTU for uranium heavy metal.
        defect_burnup = zero_safe_log_power(burnup, self.defect_burnup_exponent)
        annealing = 1.0 / (
            1.0 + self.annealing_prefactor * jnp.exp(-self.annealing_temperature / t)
        )
        phonon = 1.0 / (
            self.phonon_constant
            + self.gadolinia_coefficient * gadolinia
            + self.phonon_temperature_coefficient * t
            + self.fission_product_coefficient * burnup
            + (
                1.0
                - self.defect_recovery_fraction
                * jnp.exp(-self.defect_recovery_rate * burnup)
            )
            * self.defect_coefficient
            * defect_burnup
            * annealing
        )
        electronic = (
            self.electronic_coefficient
            / (t * t)
            * jnp.exp(-self.electronic_temperature / t)
        )
        density = 1.0 - pores
        porosity_factor = (
            self.porosity_normalization
            * density
            / (1.0 + self.pore_shape_coefficient * (1.0 - density))
        )
        return (phonon + electronic) * porosity_factor


@final
class UO2ThermalExpansion(StrictModule):
    """Solid UO2 thermal expansion strain (FRAPCON-4.0 FTHEXP).

    PNNL-19417 Rev. 2, Eq. (2.24) with the FRAPCON-4.0 UO2 constants of
    Table 2.2::

        F(T) = K1 T - K2 + K3 exp(-ED / (k T))

    (``k = 1.38e-23 J/K`` as in the source). The source defines the strain as
    "equal to zero at 300 K", while ``F(300 K) = 3.3e-4`` with the refitted
    constants; this law returns the strain relative to
    ``reference_temperature`` (default 300 K), ``F(T) - F(T_ref)``, so ``K2``
    cancels exactly and is not a parameter. The strain is linear (isotropic,
    per unit length). Solid phase only: the melting terms of Eqs. (2.25)-(2.26)
    lie outside the validity range.
    """

    linear_coefficient: Array
    defect_coefficient: Array
    defect_energy: Array
    validity: PropertyValidityRange
    reference_temperature: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        linear_coefficient: ArrayLike = 9.80e-6,
        defect_coefficient: ArrayLike = 3.16e-1,
        defect_energy: ArrayLike = 1.32e-19,
        reference_temperature: float = 300.0,
    ) -> None:
        parameters = {
            "linear_coefficient": finite_parameter(
                linear_coefficient, "linear_coefficient"
            ),
            "defect_coefficient": finite_parameter(
                defect_coefficient, "defect_coefficient"
            ),
            "defect_energy": positive_parameter(defect_energy, "defect_energy"),
        }
        reference = positive_parameter(reference_temperature, "reference_temperature")
        self.linear_coefficient = jnp.asarray(parameters["linear_coefficient"])
        self.defect_coefficient = jnp.asarray(parameters["defect_coefficient"])
        self.defect_energy = jnp.asarray(parameters["defect_energy"])
        # PNNL-19417 Rev. 2 Section 2.5.2: applicable through the melting
        # temperature of the fuel (3111.15 K for UO2); zero strain at 300 K.
        self.validity = PropertyValidityRange(
            {"temperature": (300.0, 3111.15, "K")},
            source_id="pnnl-19417-rev2:2.5.2",
        )
        self.reference_temperature = reference
        self.source_id = "pnnl-19417-rev2:eq-2.24-table-2.2"
        self.law_id = canonical_fingerprint(
            {
                "kind": "uo2-thermal-expansion-fthexp",
                "parameters": parameters,
                "reference_temperature": reference,
                "source": self.source_id,
            }
        )

    def _expansion(self, temperature: Array, /) -> Array:
        return self.linear_coefficient * temperature + self.defect_coefficient * jnp.exp(
            -self.defect_energy / (_FTHEXP_BOLTZMANN_CONSTANT * temperature)
        )

    def linear_strain(self, temperature: ArrayLike, /) -> Array:
        """Linear thermal strain [-] at temperature [K] relative to the reference."""

        t = float64_array(temperature)
        return self._expansion(t) - self._expansion(
            jnp.asarray(self.reference_temperature)
        )


@final
class UO2ElasticModuli(StrictModule):
    """Shear and bulk moduli [Pa] of isotropic UO2."""

    shear: Array
    bulk: Array


@final
class UO2Elasticity(StrictModule):
    """Isotropic elastic moduli of stoichiometric UO2 (MATPRO FELMOD, FPOIR).

    NUREG/CR-6150 Vol. 4, Eq. (2-48)::

        E = 2.334e11 [1 - 2.752 (1 - D)] [1 - 1.0915e-4 T]   [Pa]

    with ``D = 1 - porosity`` the fraction of theoretical density, and the
    temperature-independent Poisson ratio 0.316 of FPOIR (Section 2.6.5, from
    Wachtman's E = 2.30e11 Pa, G = 0.874e11 Pa). The nonstoichiometry and PuO2
    factor of Eq. (2-49) is one for stoichiometric UO2, the only fuel modeled.
    """

    reference_modulus: Array
    porosity_coefficient: Array
    temperature_coefficient: Array
    poisson: Array
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        reference_modulus: ArrayLike = 2.334e11,
        porosity_coefficient: ArrayLike = 2.752,
        temperature_coefficient: ArrayLike = 1.0915e-4,
        poisson_ratio: ArrayLike = 0.316,
    ) -> None:
        parameters = {
            "reference_modulus": positive_parameter(
                reference_modulus, "reference_modulus"
            ),
            "porosity_coefficient": finite_parameter(
                porosity_coefficient, "porosity_coefficient"
            ),
            "temperature_coefficient": finite_parameter(
                temperature_coefficient, "temperature_coefficient"
            ),
            "poisson_ratio": finite_parameter(poisson_ratio, "poisson_ratio"),
        }
        if not -1.0 < parameters["poisson_ratio"] < 0.5:
            raise ValueError("poisson_ratio must lie in (-1, 0.5).")
        self.reference_modulus = jnp.asarray(parameters["reference_modulus"])
        self.porosity_coefficient = jnp.asarray(parameters["porosity_coefficient"])
        self.temperature_coefficient = jnp.asarray(parameters["temperature_coefficient"])
        self.poisson = jnp.asarray(parameters["poisson_ratio"])
        # NUREG/CR-6150 Vol. 4 Section 2.6.3: data 300-1600 K described by the
        # linear form, extrapolated to the approximate melting point 3113 K;
        # porosity dependence fitted for porosities between 0 and 0.1.
        self.validity = PropertyValidityRange(
            {"temperature": (300.0, 3113.0, "K"), "porosity": (0.0, 0.1, "1")},
            source_id="nureg-cr-6150-vol4:2.6.3",
        )
        self.source_id = "nureg-cr-6150-vol4:felmod-eq-2-48:fpoir-2.6.5"
        self.law_id = canonical_fingerprint(
            {
                "kind": "uo2-elasticity-felmod-fpoir",
                "parameters": parameters,
                "source": self.source_id,
            }
        )

    def youngs_modulus(self, temperature: ArrayLike, porosity: ArrayLike, /) -> Array:
        """Young's modulus [Pa] at temperature [K] and porosity fraction."""

        t, pores = jnp.broadcast_arrays(
            float64_array(temperature), float64_array(porosity)
        )
        return (
            self.reference_modulus
            * (1.0 - self.porosity_coefficient * pores)
            * (1.0 - self.temperature_coefficient * t)
        )

    def poisson_ratio(self, temperature: ArrayLike, porosity: ArrayLike, /) -> Array:
        """Poisson ratio [-]; FPOIR is independent of temperature and porosity."""

        t, _ = jnp.broadcast_arrays(float64_array(temperature), float64_array(porosity))
        return jnp.broadcast_to(self.poisson, t.shape)

    def shear_modulus(self, temperature: ArrayLike, porosity: ArrayLike, /) -> Array:
        """Shear modulus ``E / (2 (1 + nu))`` [Pa]."""

        return self.youngs_modulus(temperature, porosity) / (2.0 * (1.0 + self.poisson))

    def bulk_modulus(self, temperature: ArrayLike, porosity: ArrayLike, /) -> Array:
        """Bulk modulus ``E / (3 (1 - 2 nu))`` [Pa]."""

        return self.youngs_modulus(temperature, porosity) / (
            3.0 * (1.0 - 2.0 * self.poisson)
        )

    def moduli(self, temperature: ArrayLike, porosity: ArrayLike, /) -> UO2ElasticModuli:
        """Shear ``E / (2 (1 + nu))`` and bulk ``E / (3 (1 - 2 nu))`` moduli [Pa]."""

        youngs = self.youngs_modulus(temperature, porosity)
        return UO2ElasticModuli(
            shear=youngs / (2.0 * (1.0 + self.poisson)),
            bulk=youngs / (3.0 * (1.0 - 2.0 * self.poisson)),
        )


@final
class UO2Emissivity(StrictModule):
    """Total hemispherical emissivity of UO2 (FRAPCON-4.0 FEMISS).

    PNNL-19417 Rev. 2, Eq. (2.23): ``e = 0.78557 + 1.5263e-5 T`` (T in K),
    standard error 8.9% against its database.
    """

    intercept: Array
    slope: Array
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self, /, *, intercept: ArrayLike = 0.78557, slope: ArrayLike = 1.5263e-5
    ) -> None:
        parameters = {
            "intercept": finite_parameter(intercept, "intercept"),
            "slope": finite_parameter(slope, "slope"),
        }
        self.intercept = jnp.asarray(parameters["intercept"])
        self.slope = jnp.asarray(parameters["slope"])
        # MatLib-1.2.1 (PNNL-35702) Section 2.1.5.3: applicable 300 to 2500 K.
        self.validity = PropertyValidityRange(
            {"temperature": (300.0, 2500.0, "K")},
            source_id="pnnl-35702:2.1.5.3",
        )
        self.source_id = "pnnl-19417-rev2:eq-2.23"
        self.law_id = canonical_fingerprint(
            {
                "kind": "uo2-emissivity-femiss",
                "parameters": parameters,
                "source": self.source_id,
            }
        )

    def emissivity(self, temperature: ArrayLike, /) -> Array:
        """Total hemispherical emissivity [-] at temperature [K]."""

        return self.intercept + self.slope * float64_array(temperature)


__all__ = [
    "UO2_THEORETICAL_DENSITY",
    "UO2ElasticModuli",
    "UO2Elasticity",
    "UO2Emissivity",
    "UO2HeatCapacity",
    "UO2ThermalConductivity",
    "UO2ThermalExpansion",
]
