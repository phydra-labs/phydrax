#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""FRAPCON-4.0 fuel-cladding gap conductance for UO2 fuel in Zircaloy cladding.

The gap conductance is the sum of gas conduction, solid-solid contact conduction,
and gray-body radiation (Geelhood et al., PNNL-19418 Vol. 1 Rev. 2, 2015,
Section 2.3.4, Eqs. 2.12-2.19). Gas and cladding hardness properties follow the
FRAPCON-4.0 material library (Luscher et al., PNNL-19417 Rev. 2, 2015):
GTHCON (Section 4.1) and CMHARD (Section 3.9).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._dtype_names import inexact_result_type
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...equations import brokaw_mixture_conductivity
from ...materials import PropertyValidityRange
from ._host_validation import finite_scalar, finite_vector, host_list
from ._identity import FUEL_GAS_SPECIES


# PNNL-19417 Rev. 2 Table 4.4 (GASMWT), g/mol, in FUEL_GAS_SPECIES order.
FUEL_GAS_MOLAR_MASS_G_PER_MOL: tuple[float, ...] = (
    4.0026,
    39.94766,
    83.8,
    131.3,
    28.01403,
)
# PNNL-19417 Rev. 2 Table 4.2 (FRAPCON-4.0 GTHCON), k = A T^B with T in K.
_GTHCON_COEFFICIENT: tuple[float, ...] = (
    2.531e-3,
    4.092e-4,
    1.966e-4,
    9.825e-5,
    2.984e-4,
)
_GTHCON_EXPONENT: tuple[float, ...] = (0.7146, 0.6748, 0.7006, 0.7334, 0.7799)
# PNNL-19417 Rev. 2 Eq. 3.58 (CMHARD), ln(MH / Pa) polynomial in T / K.
_CMHARD_COEFFICIENTS: tuple[float, ...] = (26.034, -2.6394e-2, 4.3502e-5, -2.5621e-8)
# 1 kgf/cm^2 in Pa (standard gravity 9.80665 m/s^2 over 1 cm^2).
_PASCAL_PER_KGF_PER_SQUARE_CENTIMETER = 98066.5
# PNNL-19418 Vol. 1 Rev. 2 Eq. 2.19: E = exp(5.738 - 0.528 ln(3.937e7 R_f)); the
# factor converts the fuel roughness from m to microinches.
_MICROINCH_PER_METER = 3.937e7
# PNNL-19418 Vol. 1 Rev. 2 Eq. 2.19 branch thresholds on the relative contact
# pressure P_rel = P / MH: above the first the contact term is linear in P_rel
# with multiplier R_mult; between the two it is the constant plateau; below
# the second it follows sqrt(P_rel). R_mult saturates above the third.
_CONTACT_LINEAR_RELATIVE_PRESSURE = 0.003
_CONTACT_SQUARE_ROOT_RELATIVE_PRESSURE = 9.0e-6
_CONTACT_MULTIPLIER_SATURATION_RELATIVE_PRESSURE = 0.0087
# PNNL-19417 Rev. 2 Appendix A.3.1: GTHCON input temperature guard [K].
_GTHCON_TEMPERATURE_RANGE_K = (0.0, 5555.0)
# PNNL-19417 Rev. 2 Section 3.9: CMHARD applicability [K].
_CMHARD_TEMPERATURE_RANGE_K = (298.0, 875.0)
# NUREG/CR-0330 Eqs. 58-60 (Ullman et al. 1974): helium and xenon
# accommodation a = a0 + a1 T with T in K, and the measured range.
_ULLMAN_HELIUM_ACCOMMODATION = (0.425, -2.3e-4)
_ULLMAN_XENON_ACCOMMODATION = (0.749, -2.5e-4)
_ULLMAN_MEASUREMENT_TEMPERATURE_RANGE_K = (500.0, 1000.0)
# NUREG/CR-0330 Sec. 5.4 GAPCON comparison: UO2/Zircaloy gap measurements at
# 293-873 K and a nominal 0.103 MPa gas pressure.
_GAPCON_COMPARISON_TEMPERATURE_RANGE_K = (293.0, 873.0)
_GAPCON_COMPARISON_PRESSURE_RANGE_PA = (0.103e6, 0.103e6)


@final
class FuelGasConductivity(StrictModule):
    """FRAPCON-4.0 GTHCON gas conductivity of the fuel-rod gas species.

    Pure species follow ``k_i = A_i T^B_i`` (W/m/K, T in K) with the FRAPCON-4.0
    coefficients of PNNL-19417 Rev. 2 Table 4.2, in ``FUEL_GAS_SPECIES`` order.
    Mixtures follow Brokaw's rule (``brokaw_mixture_conductivity``) with the
    molecular weights of PNNL-19417 Rev. 2 Table 4.4. The validity interval is the
    input guard of the GTHCON listing (PNNL-19417 Rev. 2 Appendix A.3.1).
    """

    coefficient: Array
    exponent: Array
    molar_mass_g_per_mol: tuple[float, ...] = eqx.field(static=True)
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        coefficient: ArrayLike | Sequence[float] = _GTHCON_COEFFICIENT,
        exponent: ArrayLike | Sequence[float] = _GTHCON_EXPONENT,
    ) -> None:
        species = len(FUEL_GAS_SPECIES)
        coefficient_ = finite_vector(coefficient, "coefficient", species, sign="positive")
        exponent_ = finite_vector(exponent, "exponent", species)
        source = "pnnl-19417-rev2:GTHCON"
        self.coefficient = coefficient_
        self.exponent = exponent_
        self.molar_mass_g_per_mol = FUEL_GAS_MOLAR_MASS_G_PER_MOL
        self.validity = PropertyValidityRange(
            {"temperature": (*_GTHCON_TEMPERATURE_RANGE_K, "K")}, source_id=source
        )
        self.source_id = source
        self.law_id = canonical_fingerprint(
            {
                "kind": "nuclear-fuel-gas-conductivity",
                "species": list(FUEL_GAS_SPECIES),
                "coefficient": host_list(coefficient_),
                "exponent": host_list(exponent_),
                "molar_mass_g_per_mol": list(FUEL_GAS_MOLAR_MASS_G_PER_MOL),
                "source": source,
            }
        )

    def species_conductivity(self, temperature: ArrayLike, /) -> Array:
        """Pure-species conductivities ``(..., 5)`` in W/m/K at ``temperature (...)`` K."""

        temperature_ = jnp.asarray(temperature)
        dtype = inexact_result_type(temperature_, self.coefficient)
        temperature_ = temperature_.astype(dtype)[..., None]
        parameter_shape = (1,) * (temperature_.ndim - 1) + (len(FUEL_GAS_SPECIES),)
        coefficient = self.coefficient.astype(dtype).reshape(parameter_shape)
        exponent = self.exponent.astype(dtype).reshape(parameter_shape)
        return coefficient * temperature_**exponent

    def mixture_conductivity(
        self, temperature: ArrayLike, mole_fractions: ArrayLike, /
    ) -> Array:
        """Mixture conductivity ``(...)`` in W/m/K of ``mole_fractions (..., 5)``."""

        mole = jnp.asarray(mole_fractions)
        if mole.ndim == 0 or mole.shape[-1] != len(FUEL_GAS_SPECIES):
            raise ValueError("mole_fractions must end with the fuel gas species axis.")
        species = self.species_conductivity(temperature)
        return brokaw_mixture_conductivity(
            mole, species, jnp.asarray(self.molar_mass_g_per_mol, dtype=species.dtype)
        )


@final
class ZircaloyMeyerHardness(StrictModule):
    """MATPRO/FRAPCON-4.0 CMHARD Meyer hardness of Zircaloy cladding.

    ``MH = exp(c0 + c1 T + c2 T^2 + c3 T^3)`` in Pa with ``T`` in K (PNNL-19417
    Rev. 2 Eq. 3.58, regression of Peggs and Godin 1975 data from 298 to 877 K;
    applicable up to 875 K). FRAPCON-4.0 applies no lower bound.
    """

    coefficients: Array
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self, /, *, coefficients: ArrayLike | Sequence[float] = _CMHARD_COEFFICIENTS
    ) -> None:
        coefficients_ = finite_vector(coefficients, "coefficients", 4)
        source = "pnnl-19417-rev2:CMHARD"
        self.coefficients = coefficients_
        self.validity = PropertyValidityRange(
            {"temperature": (*_CMHARD_TEMPERATURE_RANGE_K, "K")}, source_id=source
        )
        self.source_id = source
        self.law_id = canonical_fingerprint(
            {
                "kind": "zircaloy-meyer-hardness",
                "coefficients": host_list(coefficients_),
                "source": source,
            }
        )

    def hardness(self, temperature: ArrayLike, /) -> Array:
        """Meyer hardness in Pa at cladding ``temperature`` in K."""

        temperature_ = jnp.asarray(temperature)
        dtype = inexact_result_type(temperature_, self.coefficients)
        temperature_ = temperature_.astype(dtype)
        coefficients = self.coefficients.astype(dtype)
        polynomial = coefficients[0] + temperature_ * (
            coefficients[1]
            + temperature_ * (coefficients[2] + temperature_ * coefficients[3])
        )
        return jnp.exp(polynomial)


@final
class GasAccommodationLaw(StrictModule):
    """Accommodation coefficient of each gas species, linear in temperature.

    ``coefficients (5, 2)`` hold ``(a0, a1)`` of ``a = a0 + a1 T`` (T in K) per
    species in ``FUEL_GAS_SPECIES`` order. ``validity`` is the temperature range
    of the model. The range of the underlying accommodation measurements and
    the temperature/pressure range of an integral gap-conductance comparison
    are reported separately; neither is extended to other rod pressures.
    """

    coefficients: Array
    validity: PropertyValidityRange
    measurement_validity: PropertyValidityRange | None
    comparison_validity: PropertyValidityRange | None
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        coefficients: ArrayLike,
        temperature_range: tuple[float, float],
        source_id: str,
        measurement_temperature_range: tuple[float, float] | None = None,
        comparison_temperature_range: tuple[float, float] | None = None,
        comparison_pressure_range: tuple[float, float] | None = None,
    ) -> None:
        host = np.asarray(coefficients, dtype=np.float64)
        bounds = np.asarray(temperature_range, dtype=np.float64)
        if host.shape != (len(FUEL_GAS_SPECIES), 2) or not np.all(np.isfinite(host)):
            raise ValueError("coefficients must have finite shape (5, 2).")
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
            raise ValueError("temperature_range must contain two finite bounds.")
        if not 0.0 < bounds[0] < bounds[1]:
            raise ValueError("temperature_range must be positive and increasing.")
        endpoints = host[:, :1] + host[:, 1:] * bounds[None, :]
        if np.any(endpoints <= 0.0) or np.any(endpoints > 1.0):
            raise ValueError("Accommodation must lie in (0, 1] throughout its range.")
        if not isinstance(source_id, str):
            raise TypeError("source_id must be a string.")
        if not source_id.strip():
            raise ValueError("source_id must name the accommodation data source.")
        if (comparison_temperature_range is None) != (comparison_pressure_range is None):
            raise ValueError(
                "A gap comparison requires both temperature and pressure ranges."
            )
        self.coefficients = jnp.asarray(host)
        self.source_id = source_id
        self.validity = PropertyValidityRange(
            {"gas_temperature": (float(bounds[0]), float(bounds[1]), "K")},
            source_id=source_id,
        )
        self.measurement_validity = (
            None
            if measurement_temperature_range is None
            else PropertyValidityRange(
                {"gas_temperature": (*measurement_temperature_range, "K")},
                source_id=source_id,
            )
        )
        if (
            comparison_temperature_range is not None
            and comparison_pressure_range is not None
        ):
            self.comparison_validity = PropertyValidityRange(
                {
                    "gas_temperature": (*comparison_temperature_range, "K"),
                    "gas_pressure": (*comparison_pressure_range, "Pa"),
                },
                source_id=source_id,
            )
        else:
            self.comparison_validity = None
        self.law_id = canonical_fingerprint(
            {
                "kind": "linear-gas-accommodation",
                "species": list(FUEL_GAS_SPECIES),
                "coefficients": host.tolist(),
                "temperature_range": bounds.tolist(),
                "measurement_temperature_range": measurement_temperature_range,
                "comparison_temperature_range": comparison_temperature_range,
                "comparison_pressure_range": comparison_pressure_range,
                "source": source_id,
            }
        )

    @classmethod
    def ullman(cls) -> GasAccommodationLaw:
        """NUREG/CR-0330 Eqs. 58-60, valid over the measured 500-1000 K.

        Species other than helium and xenon interpolate linearly in molecular
        weight between them.
        """

        masses = np.asarray(FUEL_GAS_MOLAR_MASS_G_PER_MOL, dtype=np.float64)
        helium_index = FUEL_GAS_SPECIES.index("He")
        xenon_index = FUEL_GAS_SPECIES.index("Xe")
        weight = (masses - masses[helium_index]) / (
            masses[xenon_index] - masses[helium_index]
        )
        helium = np.asarray(_ULLMAN_HELIUM_ACCOMMODATION, dtype=np.float64)
        xenon = np.asarray(_ULLMAN_XENON_ACCOMMODATION, dtype=np.float64)
        coefficients = helium[None, :] + weight[:, None] * (xenon - helium)[None, :]
        return cls(
            coefficients=coefficients,
            temperature_range=_ULLMAN_MEASUREMENT_TEMPERATURE_RANGE_K,
            measurement_temperature_range=_ULLMAN_MEASUREMENT_TEMPERATURE_RANGE_K,
            source_id="nureg-cr-0330:ullman-accommodation-eqs58-60",
        )

    @classmethod
    def gap_comparison(cls) -> GasAccommodationLaw:
        """The Ullman fit over the 293-873 K GAPCON comparison at 0.103 MPa.

        Garnier and Begej (NUREG/CR-0330 Sec. 5.4) compare the extrapolated
        Ullman fit with UO2/Zircaloy gap-conductance measurements, not with
        direct accommodation measurements. Other rod pressures lie outside
        that comparison.
        """

        measured = cls.ullman()
        return cls(
            coefficients=measured.coefficients,
            temperature_range=_GAPCON_COMPARISON_TEMPERATURE_RANGE_K,
            measurement_temperature_range=_ULLMAN_MEASUREMENT_TEMPERATURE_RANGE_K,
            comparison_temperature_range=_GAPCON_COMPARISON_TEMPERATURE_RANGE_K,
            comparison_pressure_range=_GAPCON_COMPARISON_PRESSURE_RANGE_PA,
            source_id="nureg-cr-0330:gap-comparison",
        )

    def evaluate(self, gas_temperature: ArrayLike, /) -> Array:
        """Accommodation ``(..., 5)`` at ``gas_temperature (...)`` [K], unclipped."""

        temperature = jnp.asarray(gas_temperature)
        dtype = inexact_result_type(temperature, self.coefficients)
        coefficients = self.coefficients.astype(dtype)
        species_shape = (1,) * temperature.ndim + (len(FUEL_GAS_SPECIES),)
        return coefficients[:, 0].reshape(species_shape) + temperature.astype(dtype)[
            ..., None
        ] * coefficients[:, 1].reshape(species_shape)


@final
class GapConductance(StrictModule):
    """Gap conductance components [W/(m^2 K)] and their validity flags.

    ``effective_gap_width`` is the FRAPCON gas-conduction length ``Delta x`` [m]
    of PNNL-19418 Vol. 1 Rev. 2 Eq. 2.17; ``gas`` is meaningful only where it is
    positive. ``successful`` marks physical inputs with finite nonnegative
    conduction terms; ``within_validity`` marks gas and cladding temperatures
    inside every correlation range; ``accommodation_extrapolated`` marks gas
    temperatures outside the accommodation measurements; and
    ``within_source_comparison`` marks states inside the accommodation law's
    gap-conductance comparison range.
    """

    gas: Array
    solid: Array
    radiation: Array
    total: Array
    effective_gap_width: Array
    successful: Array
    within_validity: Array
    accommodation_extrapolated: Array
    within_source_comparison: Array


class _GapInputs(NamedTuple):
    """Elementwise inputs of `GapConductanceLaw.conductance` in SI units.

    Temperatures [K] are the fuel and cladding surface temperatures; radii [m]
    are the fuel surface and cladding bore radii.
    """

    fuel_temperature: Array
    cladding_temperature: Array
    gap_width: Array
    contact_pressure: Array
    gas_pressure: Array
    fuel_roughness: Array
    cladding_roughness: Array
    fuel_emissivity: Array
    cladding_emissivity: Array
    fuel_radius: Array
    cladding_inner_radius: Array
    fuel_conductivity: Array
    cladding_conductivity: Array


@final
class GapConductanceLaw(StrictModule):
    """FRAPCON-4.0 GAPCON-type fuel-cladding gap conductance (PNNL-19418 Vol. 1 Rev. 2).

    Gas conduction (Eqs. 2.16-2.18)::

        h_gas = k_mix(T_gas) / Delta x
        Delta x = exp(-c_P P) (R_f + R_c) + m_jump (g_f + g_c) - b + d
        g_f + g_c = A k_mix sqrt(T_gas) / P_gas / sum_i(a_i x_i / sqrt(M_i))

    with ``T_gas`` the mean of the two surface temperatures, ``P`` the contact
    pressure in kgf/cm^2 (zero for an open gap, so the closed-gap roughness factor
    reduces to the open-gap sum), ``d`` the open gap width, and ``M_i`` in g/mol.
    ``m_jump = 1.8`` is the GAPCON-THERMAL-II correction to the Lloyd jump distance.
    Accommodation coefficients follow Ullman et al. (J. Nucl. Mater. 51 (1974) 277)
    as used with this jump-distance model by Lanning and Hann (BNWL-1894, 1975)
    and reproduced by Garnier and Begej (NUREG/CR-0330, 1979, Eqs. 58-60):
    ``a_He = 0.425 - 2.3e-4 T``, ``a_Xe = 0.749 - 2.5e-4 T`` at the mean interface
    temperature, and linear interpolation in molecular weight between He and Xe
    for the other species.

    Solid contact (Eq. 2.19), with ``P_rel = P / MH(T_c)`` from CMHARD,
    ``K_m = 2 k_f k_c / (k_f + k_c)``, ``R = sqrt(R_f^2 + R_c^2)`` and
    ``E = exp(5.738 - 0.528 ln(3.937e7 R_f))``::

        h_solid = 0.4166 K_m P_rel R_mult / (R E)   if P_rel > 0.003
                = 0.00125 K_m / (R E)               if 9e-6 < P_rel <= 0.003
                = 0.4166 K_m sqrt(P_rel) / (R E)    if P_rel <= 9e-6
        R_mult  = 333.3 P_rel (P_rel <= 0.0087), 2.9 otherwise

    Published rounded coefficients introduce small jumps at the branch
    thresholds; they are not smoothed or retuned. Derivatives are branchwise,
    and the ``sqrt(P_rel)`` branch returns zero derivative at zero pressure.

    Radiation (Eqs. 2.13-2.15), written in factored form so it stays finite at
    equal surface temperatures::

        h_rad = sigma (T_f^2 + T_c^2)(T_f + T_c) / [1/e_f + (r_f/r_c)(1/e_c - 1)]

    with the FRAPCON value ``sigma = 5.6697e-8`` W/m^2/K^4.
    """

    gas_conductivity: FuelGasConductivity
    meyer_hardness: ZircaloyMeyerHardness
    jump_distance_coefficient: Array
    jump_distance_multiplier: Array
    roughness_offset: Array
    closed_gap_roughness_decay: Array
    accommodation: GasAccommodationLaw
    contact_coefficient: Array
    contact_plateau: Array
    contact_multiplier_slope: Array
    contact_multiplier_limit: Array
    roughness_exponent_coefficients: Array
    stefan_boltzmann: Array
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        gas_conductivity: FuelGasConductivity | None = None,
        meyer_hardness: ZircaloyMeyerHardness | None = None,
        jump_distance_coefficient: ArrayLike = 0.0137,
        jump_distance_multiplier: ArrayLike = 1.8,
        roughness_offset: ArrayLike = 1.397e-6,
        closed_gap_roughness_decay: ArrayLike = 0.00125,
        accommodation: GasAccommodationLaw | None = None,
        contact_coefficient: ArrayLike = 0.4166,
        contact_plateau: ArrayLike = 0.00125,
        contact_multiplier_slope: ArrayLike = 333.3,
        contact_multiplier_limit: ArrayLike = 2.9,
        roughness_exponent_coefficients: ArrayLike | Sequence[float] = (5.738, -0.528),
        stefan_boltzmann: ArrayLike = 5.6697e-8,
    ) -> None:
        gas = FuelGasConductivity() if gas_conductivity is None else gas_conductivity
        hardness = ZircaloyMeyerHardness() if meyer_hardness is None else meyer_hardness
        if not isinstance(gas, FuelGasConductivity):
            raise TypeError("gas_conductivity must be a FuelGasConductivity.")
        if not isinstance(hardness, ZircaloyMeyerHardness):
            raise TypeError("meyer_hardness must be a ZircaloyMeyerHardness.")
        accommodation_ = (
            GasAccommodationLaw.ullman() if accommodation is None else accommodation
        )
        if not isinstance(accommodation_, GasAccommodationLaw):
            raise TypeError("accommodation must be a GasAccommodationLaw.")
        scalars = {
            "jump_distance_coefficient": finite_scalar(
                jump_distance_coefficient, "jump_distance_coefficient", positive=True
            ),
            "jump_distance_multiplier": finite_scalar(
                jump_distance_multiplier, "jump_distance_multiplier", positive=True
            ),
            "roughness_offset": finite_scalar(roughness_offset, "roughness_offset"),
            "closed_gap_roughness_decay": finite_scalar(
                closed_gap_roughness_decay, "closed_gap_roughness_decay"
            ),
            "contact_coefficient": finite_scalar(
                contact_coefficient, "contact_coefficient", positive=True
            ),
            "contact_plateau": finite_scalar(
                contact_plateau, "contact_plateau", positive=True
            ),
            "contact_multiplier_slope": finite_scalar(
                contact_multiplier_slope, "contact_multiplier_slope", positive=True
            ),
            "contact_multiplier_limit": finite_scalar(
                contact_multiplier_limit, "contact_multiplier_limit", positive=True
            ),
            "stefan_boltzmann": finite_scalar(
                stefan_boltzmann, "stefan_boltzmann", positive=True
            ),
        }
        vectors = {
            "roughness_exponent_coefficients": finite_vector(
                roughness_exponent_coefficients, "roughness_exponent_coefficients", 2
            ),
        }
        source = "pnnl-19418-vol1-rev2:gap-conductance"
        self.gas_conductivity = gas
        self.meyer_hardness = hardness
        self.jump_distance_coefficient = scalars["jump_distance_coefficient"]
        self.jump_distance_multiplier = scalars["jump_distance_multiplier"]
        self.roughness_offset = scalars["roughness_offset"]
        self.closed_gap_roughness_decay = scalars["closed_gap_roughness_decay"]
        self.accommodation = accommodation_
        self.contact_coefficient = scalars["contact_coefficient"]
        self.contact_plateau = scalars["contact_plateau"]
        self.contact_multiplier_slope = scalars["contact_multiplier_slope"]
        self.contact_multiplier_limit = scalars["contact_multiplier_limit"]
        self.roughness_exponent_coefficients = vectors["roughness_exponent_coefficients"]
        self.stefan_boltzmann = scalars["stefan_boltzmann"]
        self.validity = accommodation_.validity
        self.source_id = source
        self.law_id = canonical_fingerprint(
            {
                "kind": "nuclear-fuel-gap-conductance",
                "gas_conductivity": gas.law_id,
                "meyer_hardness": hardness.law_id,
                "accommodation": accommodation_.law_id,
                "scalars": {name: host_list(value)[0] for name, value in scalars.items()},
                "vectors": {name: host_list(value) for name, value in vectors.items()},
                "source": source,
            }
        )

    def conductance(
        self,
        *,
        fuel_surface_temperature: ArrayLike,
        cladding_surface_temperature: ArrayLike,
        gap_width: ArrayLike,
        contact_pressure: ArrayLike,
        gas_mole_fractions: ArrayLike,
        gas_pressure: ArrayLike,
        fuel_roughness: ArrayLike,
        cladding_roughness: ArrayLike,
        fuel_emissivity: ArrayLike,
        cladding_emissivity: ArrayLike,
        fuel_radius: ArrayLike,
        cladding_inner_radius: ArrayLike,
        fuel_conductivity: ArrayLike,
        cladding_conductivity: ArrayLike,
    ) -> GapConductance:
        """Gap conductance of broadcast-compatible SI inputs.

        ``gap_width`` (m) is the open radial gap (zero when closed) and
        ``contact_pressure`` (Pa) the fuel-cladding interfacial pressure (zero when
        open). ``gas_mole_fractions`` ends with the ``FUEL_GAS_SPECIES`` axis.
        """

        mole_fractions = jnp.asarray(gas_mole_fractions)
        if mole_fractions.ndim == 0 or mole_fractions.shape[-1] != len(FUEL_GAS_SPECIES):
            raise ValueError(
                "gas_mole_fractions must end with the fuel gas species axis."
            )
        given = _GapInputs(
            fuel_temperature=jnp.asarray(fuel_surface_temperature),
            cladding_temperature=jnp.asarray(cladding_surface_temperature),
            gap_width=jnp.asarray(gap_width),
            contact_pressure=jnp.asarray(contact_pressure),
            gas_pressure=jnp.asarray(gas_pressure),
            fuel_roughness=jnp.asarray(fuel_roughness),
            cladding_roughness=jnp.asarray(cladding_roughness),
            fuel_emissivity=jnp.asarray(fuel_emissivity),
            cladding_emissivity=jnp.asarray(cladding_emissivity),
            fuel_radius=jnp.asarray(fuel_radius),
            cladding_inner_radius=jnp.asarray(cladding_inner_radius),
            fuel_conductivity=jnp.asarray(fuel_conductivity),
            cladding_conductivity=jnp.asarray(cladding_conductivity),
        )
        dtype = inexact_result_type(
            mole_fractions, *given, self.jump_distance_coefficient
        )
        shape = jnp.broadcast_shapes(
            mole_fractions.shape[:-1], *(value.shape for value in given)
        )
        mole_fractions = jnp.broadcast_to(
            mole_fractions.astype(dtype), (*shape, mole_fractions.shape[-1])
        )
        inputs = given._make(
            jnp.broadcast_to(value.astype(dtype), shape) for value in given
        )
        gas_temperature = 0.5 * (inputs.fuel_temperature + inputs.cladding_temperature)
        gas, effective_gap_width = self._gas_conductance(
            gas_temperature, mole_fractions, inputs
        )
        solid = self._solid_conductance(inputs)
        radiation = self._radiation_conductance(inputs)
        total = gas + solid + radiation
        within_validity = (
            self.validity.contains(gas_temperature=gas_temperature)
            & self.gas_conductivity.validity.contains(temperature=gas_temperature)
            & self.meyer_hardness.validity.contains(
                temperature=inputs.cladding_temperature
            )
        )
        measurement = self.accommodation.measurement_validity
        accommodation_extrapolated = (
            jnp.ones(shape, dtype=jnp.bool_)
            if measurement is None
            else ~measurement.contains(gas_temperature=gas_temperature)
        )
        comparison = self.accommodation.comparison_validity
        within_source_comparison = (
            jnp.zeros(shape, dtype=jnp.bool_)
            if comparison is None
            else comparison.contains(
                gas_temperature=gas_temperature, gas_pressure=inputs.gas_pressure
            )
        )
        return GapConductance(
            gas=gas,
            solid=solid,
            radiation=radiation,
            total=total,
            effective_gap_width=effective_gap_width,
            successful=_physical_inputs(inputs, effective_gap_width)
            & jnp.isfinite(total)
            & (gas >= 0.0)
            & (solid >= 0.0),
            within_validity=within_validity,
            accommodation_extrapolated=accommodation_extrapolated,
            within_source_comparison=within_source_comparison,
        )

    def _gas_conductance(
        self, gas_temperature: Array, mole_fractions: Array, inputs: _GapInputs, /
    ) -> tuple[Array, Array]:
        """Gas conduction and its effective gap width (Eqs. 2.16-2.18)."""

        dtype = gas_temperature.dtype
        conductivity = self.gas_conductivity.mixture_conductivity(
            gas_temperature, mole_fractions
        )
        species_shape = (1,) * gas_temperature.ndim + (len(FUEL_GAS_SPECIES),)
        masses = jnp.asarray(
            self.gas_conductivity.molar_mass_g_per_mol, dtype=dtype
        ).reshape(species_shape)
        accommodation = self.accommodation.evaluate(gas_temperature)
        accommodation_sum = jnp.sum(
            accommodation * mole_fractions / jnp.sqrt(masses), axis=-1
        )
        jump_distance = (
            self.jump_distance_coefficient.astype(dtype)
            * conductivity
            * jnp.sqrt(gas_temperature)
            / inputs.gas_pressure
            / accommodation_sum
        )
        contact_pressure_kgf_per_square_centimeter = (
            inputs.contact_pressure / _PASCAL_PER_KGF_PER_SQUARE_CENTIMETER
        )
        roughness = jnp.exp(
            -self.closed_gap_roughness_decay.astype(dtype)
            * contact_pressure_kgf_per_square_centimeter
        ) * (inputs.fuel_roughness + inputs.cladding_roughness)
        effective_gap_width = (
            roughness
            + self.jump_distance_multiplier.astype(dtype) * jump_distance
            - self.roughness_offset.astype(dtype)
            + inputs.gap_width
        )
        return conductivity / effective_gap_width, effective_gap_width

    def _solid_conductance(self, inputs: _GapInputs, /) -> Array:
        """Solid-solid contact conduction (Eq. 2.19)."""

        dtype = inputs.cladding_temperature.dtype
        relative_pressure = inputs.contact_pressure / self.meyer_hardness.hardness(
            inputs.cladding_temperature
        )
        mean_conductivity = (
            2.0
            * inputs.fuel_conductivity
            * inputs.cladding_conductivity
            / (inputs.fuel_conductivity + inputs.cladding_conductivity)
        )
        roughness = jnp.sqrt(inputs.fuel_roughness**2 + inputs.cladding_roughness**2)
        exponent = self.roughness_exponent_coefficients.astype(dtype)
        roughness_factor = jnp.exp(
            exponent[0]
            + exponent[1] * jnp.log(_MICROINCH_PER_METER * inputs.fuel_roughness)
        )
        scale = mean_conductivity / (roughness * roughness_factor)
        coefficient = self.contact_coefficient.astype(dtype)
        multiplier = jnp.where(
            relative_pressure <= _CONTACT_MULTIPLIER_SATURATION_RELATIVE_PRESSURE,
            self.contact_multiplier_slope.astype(dtype) * relative_pressure,
            self.contact_multiplier_limit.astype(dtype),
        )
        # The inner selection keeps the sqrt derivative finite at zero pressure.
        positive = relative_pressure > 0.0
        square_root = jnp.where(
            positive, jnp.sqrt(jnp.where(positive, relative_pressure, 1.0)), 0.0
        )
        return scale * jnp.where(
            relative_pressure > _CONTACT_LINEAR_RELATIVE_PRESSURE,
            coefficient * relative_pressure * multiplier,
            jnp.where(
                relative_pressure > _CONTACT_SQUARE_ROOT_RELATIVE_PRESSURE,
                self.contact_plateau.astype(dtype),
                coefficient * square_root,
            ),
        )

    def _radiation_conductance(self, inputs: _GapInputs, /) -> Array:
        """Gray-body radiation in factored form (Eqs. 2.13-2.15)."""

        emissivity_factor = 1.0 / (
            1.0 / inputs.fuel_emissivity
            + inputs.fuel_radius
            / inputs.cladding_inner_radius
            * (1.0 / inputs.cladding_emissivity - 1.0)
        )
        return (
            self.stefan_boltzmann.astype(inputs.fuel_temperature.dtype)
            * emissivity_factor
            * (inputs.fuel_temperature**2 + inputs.cladding_temperature**2)
            * (inputs.fuel_temperature + inputs.cladding_temperature)
        )


def _physical_inputs(inputs: _GapInputs, effective_gap_width: Array, /) -> Array:
    """Whether every input lies in its physical domain and ``Delta x > 0``."""

    return (
        (inputs.fuel_temperature > 0.0)
        & (inputs.cladding_temperature > 0.0)
        & (inputs.gap_width >= 0.0)
        & (inputs.contact_pressure >= 0.0)
        & (inputs.gas_pressure > 0.0)
        & (inputs.fuel_roughness > 0.0)
        & (inputs.cladding_roughness > 0.0)
        & (inputs.fuel_emissivity > 0.0)
        & (inputs.fuel_emissivity <= 1.0)
        & (inputs.cladding_emissivity > 0.0)
        & (inputs.cladding_emissivity <= 1.0)
        & (inputs.fuel_radius > 0.0)
        & (inputs.cladding_inner_radius >= inputs.fuel_radius)
        & (inputs.fuel_conductivity > 0.0)
        & (inputs.cladding_conductivity > 0.0)
        & (effective_gap_width > 0.0)
    )


__all__ = [
    "FUEL_GAS_MOLAR_MASS_G_PER_MOL",
    "FuelGasConductivity",
    "GasAccommodationLaw",
    "GapConductance",
    "GapConductanceLaw",
    "ZircaloyMeyerHardness",
]
