#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Liquid water from IAPWS-IF97 regions 1 and 4 with IAPWS industrial transport.

Thermodynamics follow the IAPWS Revised Release on the IAPWS Industrial
Formulation 1997 for the Thermodynamic Properties of Water and Steam,
IAPWS R7-97(2012): region 1 basic equation, Eq. (7) with Tables 2-4; region 1
backward equation T(p, h), Eq. (11) with Table 6; region 4 saturation
equations, Eqs. (30) and (31) with Table 34.

Viscosity is IAPWS R12-08, Eq. (10), in its recommended industrial form
(Section 3): the critical enhancement is set to ``mu_2 = 1`` and the density is
taken from IAPWS-IF97. Section 3 permits this for states in the IAPWS-IF97
range of validity outside the near-critical region of Eq. (13)
(645.91 K < T < 650.77 K, 245.8 kg/m^3 < rho < 405.3 kg/m^3), which region 1
(T <= 623.15 K) never enters.

Thermal conductivity is IAPWS R15-11, Eq. (15), in the industrial formulation
of Section 3.1 *including* the critical enhancement ``lambda_2`` of
Eqs. (18)-(24): c_p, c_p/c_v and (d rho / d p)_T come from IAPWS-IF97 region 1,
the viscosity in Eq. (18) from the R12-08 industrial form, and
zeta(T_R, rho) from Eq. (25) with Table 6. The Section 3.1 footnote-2 guards
(zeta or c_p negative or above 1e13) concern states very close to the critical
point; zeta and c_p are finite and positive everywhere in region 1, so they
never apply here.

All functions are pure JAX, broadcast elementwise, and return the formula value
outside the validity range; `iapws97_region1_contains` reports membership.
"""

from __future__ import annotations

from typing import final, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._dtype_names import inexact_result_type
from .._strict import StrictModule
from ..ein import contract
from ._validity import PropertyValidityRange


# IAPWS R7-97(2012), Eq. (1): specific gas constant of IAPWS-IF97.
_IF97_GAS_CONSTANT = 461.526  # J/(kg K)

# Region 1 reducing constants, Eq. (7): p* = 16.53 MPa, T* = 1386 K.
_REGION1_PRESSURE = 16.53e6
_REGION1_TEMPERATURE = 1386.0

# Table 2: exponents I_i, J_i and coefficients n_i of Eq. (7).
_REGION1_I = np.asarray(
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2)
    + (2, 2, 3, 3, 3, 4, 4, 4, 5, 8, 8, 21, 23, 29, 30, 31, 32),
    dtype=np.float64,
)
_REGION1_J = np.asarray(
    (-2, -1, 0, 1, 2, 3, 4, 5, -9, -7, -1, 0, 1, 3, -3, 0, 1)
    + (3, 17, -4, 0, 6, -5, -2, 10, -8, -11, -6, -29, -31, -38, -39, -40, -41),
    dtype=np.float64,
)
_REGION1_N = np.asarray(
    (
        0.14632971213167,
        -0.84548187169114,
        -0.37563603672040e1,
        0.33855169168385e1,
        -0.95791963387872,
        0.15772038513228,
        -0.16616417199501e-1,
        0.81214629983568e-3,
        0.28319080123804e-3,
        -0.60706301565874e-3,
        -0.18990068218419e-1,
        -0.32529748770505e-1,
        -0.21841717175414e-1,
        -0.52838357969930e-4,
        -0.47184321073267e-3,
        -0.30001780793026e-3,
        0.47661393906987e-4,
        -0.44141845330846e-5,
        -0.72694996297594e-15,
        -0.31679644845054e-4,
        -0.28270797985312e-5,
        -0.85205128120103e-9,
        -0.22425281908000e-5,
        -0.65171222895601e-6,
        -0.14341729937924e-12,
        -0.40516996860117e-6,
        -0.12734301741641e-8,
        -0.17424871230634e-9,
        -0.68762131295531e-18,
        0.14478307828521e-19,
        0.26335781662795e-22,
        -0.11947622640071e-22,
        0.18228094581404e-23,
        -0.93537087292458e-25,
    ),
    dtype=np.float64,
)

# Region 1 backward equation T(p, h), Eq. (11): T* = 1 K, p* = 1 MPa,
# h* = 2500 kJ/kg; Table 6 exponents and coefficients.
_BACKWARD_PRESSURE = 1.0e6
_BACKWARD_ENTHALPY = 2.5e6
_BACKWARD_I = np.asarray(
    (0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6), dtype=np.float64
)
_BACKWARD_J = np.asarray(
    (0, 1, 2, 6, 22, 32, 0, 1, 2, 3, 4, 10, 32, 10, 32, 10, 32, 32, 32, 32),
    dtype=np.float64,
)
_BACKWARD_N = np.asarray(
    (
        -0.23872489924521e3,
        0.40421188637945e3,
        0.11349746881718e3,
        -0.58457616048039e1,
        -0.15285482413140e-3,
        -0.10866707695377e-5,
        -0.13391744872602e2,
        0.43211039183559e2,
        -0.54010067170506e2,
        0.30535892203916e2,
        -0.65964749423638e1,
        0.93965400878363e-2,
        0.11573647505340e-6,
        -0.25858641282073e-4,
        -0.40644363084799e-8,
        0.66456186191635e-7,
        0.80670734103027e-10,
        -0.93477771213947e-12,
        0.58265442020601e-14,
        -0.15020185953503e-16,
    ),
    dtype=np.float64,
)

# Region 4, Table 34 (n_1 ... n_10 of Eqs. (29)-(31)); p* = 1 MPa, T* = 1 K.
_SATURATION_PRESSURE_UNIT = 1.0e6
_SAT_N1 = 0.11670521452767e4
_SAT_N2 = -0.72421316703206e6
_SAT_N3 = -0.17073846940092e2
_SAT_N4 = 0.12020824702470e5
_SAT_N5 = -0.32325550322333e7
_SAT_N6 = 0.14915108613530e2
_SAT_N7 = -0.48232657361591e4
_SAT_N8 = 0.40511340542057e6
_SAT_N9 = -0.23855557567849
_SAT_N10 = 0.65017534844798e3

# Region 1 range of validity (R7-97(2012), Section 5.1): 273.15 K <= T <= 623.15 K,
# p_s(T) <= p <= 100 MPa. 611.213 Pa is p_s(273.15 K) (Section 8.2).
_REGION1_MINIMUM_TEMPERATURE = 273.15
_REGION1_MAXIMUM_TEMPERATURE = 623.15
_REGION1_MINIMUM_PRESSURE = 611.213
_REGION1_MAXIMUM_PRESSURE = 100.0e6

# IAPWS R12-08 / R15-11 reference constants (Eqs. (1)-(4) and (1)-(6)).
_CRITICAL_TEMPERATURE = 647.096  # K
_CRITICAL_DENSITY = 322.0  # kg/m^3
_CRITICAL_PRESSURE = 22.064e6  # Pa
_VISCOSITY_UNIT = 1.0e-6  # Pa s
_CONDUCTIVITY_UNIT = 1.0e-3  # W/(m K)
# R15-11 Eq. (6): the IAPWS-95 specific gas constant, used to reduce c_p.
_IAPWS95_GAS_CONSTANT = 461.51805  # J/(kg K)

# R12-08 Table 1 (H_i of Eq. (11)) and Table 2 (H_ij of Eq. (12), i = 0..5 rows,
# j = 0..6 columns; omitted entries are identically zero).
_VISCOSITY_H0 = np.asarray((1.67752, 2.20462, 0.6366564, -0.241605), dtype=np.float64)
_VISCOSITY_H1 = np.asarray(
    (
        (5.20094e-1, 2.22531e-1, -2.81378e-1, 1.61913e-1, -3.25372e-2, 0.0, 0.0),
        (8.50895e-2, 9.99115e-1, -9.06851e-1, 2.57399e-1, 0.0, 0.0, 0.0),
        (-1.08374, 1.88797, -7.72479e-1, 0.0, 0.0, 0.0, 0.0),
        (-2.89555e-1, 1.26613, -4.89837e-1, 0.0, 6.98452e-2, 0.0, -4.35673e-3),
        (0.0, 0.0, -2.57040e-1, 0.0, 0.0, 8.72102e-3, 0.0),
        (0.0, 1.20573e-1, 0.0, 0.0, 0.0, 0.0, -5.93264e-4),
    ),
    dtype=np.float64,
)

# R15-11 Table 1 (L_k of Eq. (16)) and Table 2 (L_ij of Eq. (17), i = 0..4 rows,
# j = 0..5 columns).
_CONDUCTIVITY_L0 = np.asarray(
    (2.443221e-3, 1.323095e-2, 6.770357e-3, -3.454586e-3, 4.096266e-4),
    dtype=np.float64,
)
_CONDUCTIVITY_L1 = np.asarray(
    (
        (
            1.60397357,
            -0.646013523,
            0.111443906,
            0.102997357,
            -0.0504123634,
            0.00609859258,
        ),
        (2.33771842, -2.78843778, 1.53616167, -0.463045512, 0.0832827019, -0.00719201245),
        (2.19650529, -4.54580785, 3.55777244, -1.40944978, 0.275418278, -0.0205938816),
        (-1.21051378, 1.60812989, -0.621178141, 0.0716373224, 0.0, 0.0),
        (-2.7203370, 4.57586331, -3.18369245, 1.1168348, -0.19268305, 0.012913842),
    ),
    dtype=np.float64,
)

# R15-11 Table 3 critical-region constants.
_CRITICAL_LAMBDA = 177.8514
_CRITICAL_WAVE_LENGTH = 0.40  # q_D^-1, nm
_CRITICAL_NU = 0.630
_CRITICAL_GAMMA = 1.239
_CRITICAL_XI0 = 0.13  # nm
_CRITICAL_GAMMA0 = 0.06
_CRITICAL_REFERENCE_TEMPERATURE = 1.5  # reduced T_R
# Eq. (21): Z(y) = 0 for y < 1.2e-7.
_CRITICAL_Z_THRESHOLD = 1.2e-7

# R15-11 Table 6 (A_ij of Eq. (25), i = 0..5 rows, j = 0..4 density ranges) and
# the reduced-density range limits of Eq. (26).
_ZETA_REFERENCE_A = np.asarray(
    (
        (
            6.53786807199516,
            6.52717759281799,
            5.35500529896124,
            1.55225959906681,
            1.11999926419994,
        ),
        (
            -5.61149954923348,
            -6.30816983387575,
            -3.96415689925446,
            0.464621290821181,
            0.595748562571649,
        ),
        (
            3.39624167361325,
            8.08379285492595,
            8.91990208918795,
            8.93237374861479,
            9.88952565078920,
        ),
        (
            -2.27492629730878,
            -9.82240510197603,
            -12.0338729505790,
            -11.0321960061126,
            -10.3255051147040,
        ),
        (
            10.2631854662709,
            12.1358413791395,
            9.19494865194302,
            6.16780999933360,
            4.66861294457414,
        ),
        (
            1.97815050331519,
            -5.54349664571295,
            -2.16866274479712,
            -0.965458722086812,
            -0.503243546373828,
        ),
    ),
    dtype=np.float64,
)
_ZETA_REFERENCE_BREAKS = np.asarray(
    (0.310559006, 0.776397516, 1.242236025, 1.863354037), dtype=np.float64
)


@final
class WaterLiquidState(StrictModule):
    """Liquid-water properties at one IAPWS-IF97 region 1 state, SI units.

    Fields broadcast with the ``(temperature, pressure)`` inputs:
    ``density`` [kg/m^3], ``specific_enthalpy`` [J/kg], ``isobaric_heat_capacity``
    [J/(kg K)], ``dynamic_viscosity`` [Pa s], ``thermal_conductivity`` [W/(m K)].
    """

    density: Array
    specific_enthalpy: Array
    isobaric_heat_capacity: Array
    dynamic_viscosity: Array
    thermal_conductivity: Array


def _real_arrays(*values: ArrayLike) -> tuple[Array, ...]:
    """Convert once to one real inexact dtype and broadcast explicitly."""

    arrays = tuple(jnp.asarray(value) for value in values)
    dtype = inexact_result_type(*arrays)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        raise TypeError("Water property inputs must be real-valued.")
    return tuple(jnp.broadcast_arrays(*(array.astype(dtype) for array in arrays)))


def _coefficients(values: np.ndarray, reference: Array, /) -> Array:
    return jnp.asarray(values, dtype=reference.dtype)


def _row(values: np.ndarray, reference: Array, /) -> Array:
    """Coefficients on a trailing term axis, rank-aligned with ``reference[..., None]``."""

    return _coefficients(values, reference).reshape((1,) * reference.ndim + (-1,))


def _powers(base: Array, exponents: np.ndarray, /) -> Array:
    """``base[..., None] ** exponents`` on a trailing term axis."""

    return base[..., None] ** _row(exponents, base)


class _Region1Gibbs(NamedTuple):
    """gamma(pi, tau) derivatives of Eq. (7), Table 4, at one state."""

    gamma_pi: Array
    gamma_pipi: Array
    gamma_tau: Array
    gamma_tautau: Array
    gamma_pitau: Array


def _region1_gibbs(temperature: Array, pressure: Array, /) -> _Region1Gibbs:
    coefficient = _coefficients(_REGION1_N, temperature)
    # 7.1 - pi > 0 and tau - 1.222 > 0 throughout region 1 (pi <= 6.05,
    # tau >= 2.224), so every real power below is well defined and smooth.
    a = 7.1 - pressure / _REGION1_PRESSURE
    b = _REGION1_TEMPERATURE / temperature - 1.222
    exponent_i = _row(_REGION1_I, a)
    exponent_j = _row(_REGION1_J, b)
    a_i = _powers(a, _REGION1_I)
    b_j = _powers(b, _REGION1_J)
    di = exponent_i * _powers(a, _REGION1_I - 1.0)
    dj = exponent_j * _powers(b, _REGION1_J - 1.0)
    dii = exponent_i * (exponent_i - 1.0) * _powers(a, _REGION1_I - 2.0)
    djj = exponent_j * (exponent_j - 1.0) * _powers(b, _REGION1_J - 2.0)
    return _Region1Gibbs(
        gamma_pi=-((di * b_j) @ coefficient),
        gamma_pipi=(dii * b_j) @ coefficient,
        gamma_tau=(a_i * dj) @ coefficient,
        gamma_tautau=(a_i * djj) @ coefficient,
        gamma_pitau=-((di * dj) @ coefficient),
    )


def _saturation_pressure(temperature: Array, /) -> Array:
    theta = temperature + _SAT_N9 / (temperature - _SAT_N10)
    a = theta * theta + _SAT_N1 * theta + _SAT_N2
    b = _SAT_N3 * theta * theta + _SAT_N4 * theta + _SAT_N5
    c = _SAT_N6 * theta * theta + _SAT_N7 * theta + _SAT_N8
    reduced = 2.0 * c / (-b + jnp.sqrt(b * b - 4.0 * a * c))
    return _SATURATION_PRESSURE_UNIT * reduced**4


def _saturation_temperature(pressure: Array, /) -> Array:
    beta = (pressure / _SATURATION_PRESSURE_UNIT) ** 0.25
    e = beta * beta + _SAT_N3 * beta + _SAT_N6
    f = _SAT_N1 * beta * beta + _SAT_N4 * beta + _SAT_N7
    g = _SAT_N2 * beta * beta + _SAT_N5 * beta + _SAT_N8
    d = 2.0 * g / (-f - jnp.sqrt(f * f - 4.0 * e * g))
    shifted = _SAT_N10 + d
    return 0.5 * (shifted - jnp.sqrt(shifted * shifted - 4.0 * (_SAT_N9 + _SAT_N10 * d)))


def _region1_enthalpy(temperature: Array, pressure: Array, /) -> Array:
    gibbs = _region1_gibbs(temperature, pressure)
    return _IF97_GAS_CONSTANT * _REGION1_TEMPERATURE * gibbs.gamma_tau


def _viscosity(temperature: Array, density: Array, /) -> Array:
    reduced_t = temperature / _CRITICAL_TEMPERATURE
    reduced_rho = density / _CRITICAL_DENSITY
    dilute_powers = _powers(reduced_t, -np.arange(4, dtype=np.float64))
    dilute = (
        100.0
        * jnp.sqrt(reduced_t)
        / (dilute_powers @ _coefficients(_VISCOSITY_H0, temperature))
    )
    t_powers = _powers(1.0 / reduced_t - 1.0, np.arange(6, dtype=np.float64))
    rho_powers = _powers(reduced_rho - 1.0, np.arange(7, dtype=np.float64))
    residual = jnp.exp(
        reduced_rho
        * contract(
            "...i,ij,...j->...",
            t_powers,
            _coefficients(_VISCOSITY_H1, temperature),
            rho_powers,
        )
    )
    # Industrial form (R12-08 Section 3): mu_2 = 1.
    return _VISCOSITY_UNIT * dilute * residual


def _zeta_reference(reduced_rho: Array, /) -> Array:
    """zeta(T_R, rho) of R15-11 Eq. (25) with the density ranges of Eq. (26)."""

    # side="left" gives j with break_{j-1} < rho <= break_j, the Eq. (26) intervals.
    column = jnp.searchsorted(
        _coefficients(_ZETA_REFERENCE_BREAKS, reduced_rho), reduced_rho, side="left"
    )
    table = _coefficients(_ZETA_REFERENCE_A, reduced_rho)
    selected = jnp.moveaxis(table[:, column], 0, -1)
    powers = _powers(reduced_rho, np.arange(6, dtype=np.float64))
    return 1.0 / jnp.sum(selected * powers, axis=-1)


def _critical_enhancement(
    reduced_t: Array,
    reduced_rho: Array,
    reduced_cp: Array,
    heat_capacity_ratio: Array,
    reduced_zeta: Array,
    reduced_viscosity: Array,
    /,
) -> Array:
    """lambda_2 (reduced) of R15-11 Eqs. (18)-(24)."""

    delta_chi = reduced_rho * (
        reduced_zeta
        - _zeta_reference(reduced_rho) * _CRITICAL_REFERENCE_TEMPERATURE / reduced_t
    )
    # Eq. (23) note: delta_chi < 0 is set to zero, giving xi = 0 and Z = 0. The
    # safe substitutes keep the unused branches finite for differentiation.
    positive = delta_chi > 0.0
    safe_chi = jnp.where(positive, delta_chi, 1.0)
    xi = jnp.where(
        positive,
        _CRITICAL_XI0 * (safe_chi / _CRITICAL_GAMMA0) ** (_CRITICAL_NU / _CRITICAL_GAMMA),
        0.0,
    )
    y = xi / _CRITICAL_WAVE_LENGTH
    active = y >= _CRITICAL_Z_THRESHOLD
    safe_y = jnp.where(active, y, 1.0)
    inverse_kappa = 1.0 / heat_capacity_ratio
    z = (
        2.0
        / (jnp.pi * safe_y)
        * (
            (1.0 - inverse_kappa) * jnp.arctan(safe_y)
            + inverse_kappa * safe_y
            - (
                1.0
                - jnp.exp(
                    -1.0
                    / (1.0 / safe_y + safe_y * safe_y / (3.0 * reduced_rho * reduced_rho))
                )
            )
        )
    )
    z = jnp.where(active, z, 0.0)
    return _CRITICAL_LAMBDA * reduced_rho * reduced_cp * reduced_t / reduced_viscosity * z


def _thermal_conductivity(
    temperature: Array,
    density: Array,
    isobaric_heat_capacity: Array,
    isochoric_heat_capacity: Array,
    density_pressure_derivative: Array,
    viscosity: Array,
    /,
) -> Array:
    reduced_t = temperature / _CRITICAL_TEMPERATURE
    reduced_rho = density / _CRITICAL_DENSITY
    dilute_powers = _powers(reduced_t, -np.arange(5, dtype=np.float64))
    dilute = jnp.sqrt(reduced_t) / (
        dilute_powers @ _coefficients(_CONDUCTIVITY_L0, temperature)
    )
    t_powers = _powers(1.0 / reduced_t - 1.0, np.arange(5, dtype=np.float64))
    rho_powers = _powers(reduced_rho - 1.0, np.arange(6, dtype=np.float64))
    residual = jnp.exp(
        reduced_rho
        * contract(
            "...i,ij,...j->...",
            t_powers,
            _coefficients(_CONDUCTIVITY_L1, temperature),
            rho_powers,
        )
    )
    enhancement = _critical_enhancement(
        reduced_t,
        reduced_rho,
        isobaric_heat_capacity / _IAPWS95_GAS_CONSTANT,
        isobaric_heat_capacity / isochoric_heat_capacity,
        density_pressure_derivative * _CRITICAL_PRESSURE / _CRITICAL_DENSITY,
        viscosity / _VISCOSITY_UNIT,
    )
    return _CONDUCTIVITY_UNIT * (dilute * residual + enhancement)


def iapws97_liquid_state(temperature: ArrayLike, pressure: ArrayLike) -> WaterLiquidState:
    """Liquid-water properties at ``temperature`` [K] and ``pressure`` [Pa].

    IAPWS-IF97 region 1 (R7-97(2012) Eq. (7), Table 3 relations) supplies
    density, enthalpy, c_p, c_v and (d rho/d p)_T; viscosity is IAPWS R12-08
    Eq. (10) with mu_2 = 1 and thermal conductivity IAPWS R15-11 Eq. (15) in the
    Section 3.1 industrial form with critical enhancement.
    """

    t, p = _real_arrays(temperature, pressure)
    gibbs = _region1_gibbs(t, p)
    tau = _REGION1_TEMPERATURE / t
    specific_volume = _IF97_GAS_CONSTANT * t * gibbs.gamma_pi / _REGION1_PRESSURE
    density = 1.0 / specific_volume
    enthalpy = _IF97_GAS_CONSTANT * _REGION1_TEMPERATURE * gibbs.gamma_tau
    cp = -_IF97_GAS_CONSTANT * tau * tau * gibbs.gamma_tautau
    coupling = gibbs.gamma_pi - tau * gibbs.gamma_pitau
    cv = _IF97_GAS_CONSTANT * (
        -tau * tau * gibbs.gamma_tautau + coupling * coupling / gibbs.gamma_pipi
    )
    # (d v / d p)_T = R T gamma_pipi / p*^2 and (d rho / d p)_T = -rho^2 (d v / d p)_T.
    density_pressure_derivative = (
        -density
        * density
        * _IF97_GAS_CONSTANT
        * t
        * gibbs.gamma_pipi
        / (_REGION1_PRESSURE * _REGION1_PRESSURE)
    )
    viscosity = _viscosity(t, density)
    conductivity = _thermal_conductivity(
        t, density, cp, cv, density_pressure_derivative, viscosity
    )
    return WaterLiquidState(
        density=density,
        specific_enthalpy=enthalpy,
        isobaric_heat_capacity=cp,
        dynamic_viscosity=viscosity,
        thermal_conductivity=conductivity,
    )


def iapws97_liquid_enthalpy(temperature: ArrayLike, pressure: ArrayLike) -> Array:
    """Region 1 specific enthalpy [J/kg] at ``temperature`` [K], ``pressure`` [Pa]."""

    t, p = _real_arrays(temperature, pressure)
    return _region1_enthalpy(t, p)


def iapws97_saturation_pressure(temperature: ArrayLike) -> Array:
    """Saturation pressure [Pa] from R7-97(2012) Eq. (30), 273.15-647.096 K."""

    (t,) = _real_arrays(temperature)
    return _saturation_pressure(t)


def iapws97_saturation_temperature(pressure: ArrayLike) -> Array:
    """Saturation temperature [K] from R7-97(2012) Eq. (31), 611.213 Pa-22.064 MPa."""

    (p,) = _real_arrays(pressure)
    return _saturation_temperature(p)


def iapws97_liquid_temperature(
    pressure: ArrayLike, specific_enthalpy: ArrayLike
) -> Array:
    """Region 1 backward temperature T(p, h) [K] from R7-97(2012) Eq. (11).

    Consistent with Eq. (7) within |dT| <= 25 mK (Eq. (12)); not valid for
    metastable superheated liquid.
    """

    p, h = _real_arrays(pressure, specific_enthalpy)
    terms = _powers(p / _BACKWARD_PRESSURE, _BACKWARD_I) * _powers(
        h / _BACKWARD_ENTHALPY + 1.0, _BACKWARD_J
    )
    return terms @ _coefficients(_BACKWARD_N, p)


def iapws2008_viscosity(temperature: ArrayLike, density: ArrayLike) -> Array:
    """IAPWS R12-08 Eq. (10) viscosity [Pa s] at ``temperature`` [K], ``density``.

    Industrial form of R12-08 Section 3 (critical enhancement mu_2 = 1),
    appropriate outside the near-critical region of R12-08 Eq. (13).
    """

    t, rho = _real_arrays(temperature, density)
    return _viscosity(t, rho)


def iapws97_region1_contains(temperature: ArrayLike, pressure: ArrayLike) -> Array:
    """Elementwise membership in IAPWS-IF97 region 1.

    Region 1 is 273.15 K <= T <= 623.15 K and p_s(T) <= p <= 100 MPa
    (R7-97(2012) Section 5.1), with p_s from Eq. (30).
    """

    t, p = _real_arrays(temperature, pressure)
    return (
        (t >= _REGION1_MINIMUM_TEMPERATURE)
        & (t <= _REGION1_MAXIMUM_TEMPERATURE)
        & (p >= _saturation_pressure(t))
        & (p <= _REGION1_MAXIMUM_PRESSURE)
    )


IAPWS97_LIQUID_RANGE = PropertyValidityRange(
    {
        "temperature": (
            _REGION1_MINIMUM_TEMPERATURE,
            _REGION1_MAXIMUM_TEMPERATURE,
            "K",
        ),
        "pressure": (_REGION1_MINIMUM_PRESSURE, _REGION1_MAXIMUM_PRESSURE, "Pa"),
    },
    source_id="iapws-r7-97-2012:region-1",
)
"""Rectangular envelope of IAPWS-IF97 region 1.

The saturation-dependent lower pressure bound p >= p_s(T) is not rectangular;
`iapws97_region1_contains` evaluates the exact region.
"""


__all__ = [
    "IAPWS97_LIQUID_RANGE",
    "WaterLiquidState",
    "iapws2008_viscosity",
    "iapws97_liquid_enthalpy",
    "iapws97_liquid_state",
    "iapws97_liquid_temperature",
    "iapws97_region1_contains",
    "iapws97_saturation_pressure",
    "iapws97_saturation_temperature",
]
