#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Creep, plasticity, and irradiation growth of stress-relief-annealed Zircaloy-4.

Every law evaluates its published correlation in the source variables and
converts at the boundary to SI: K, Pa, m, s; fast neutron flux in n/(m^2 s)
and fluence in n/m^2 (E > 1 MeV). Sources:

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
- PNNL-17700: Geelhood, Beyer and Luscher, "PNNL Stress/Strain Correlation for
  Zircaloy", 2008.

Out-of-range evaluation returns the formula value; each ``validity`` states the
fitted domain and the caller decides.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...materials import PropertyValidityRange
from ...nonlinear import LocalRootPlan
from ._cladding_properties import (
    CladdingPrincipalStrains,
    ZIRCALOY_ALPHA_PHASE_MAXIMUM_TEMPERATURE,
    ZircaloyElasticity,
    ZircaloyElasticModuli,
    ZircaloyTexture,
)
from ._constants import PASCAL_PER_MEGAPASCAL, SECONDS_PER_HOUR
from ._identity import CladdingAlloy, CladdingHeatTreatment
from ._material_math import von_mises_equivalent, zero_safe_power
from ._material_parameters import (
    finite_parameter,
    float64_array,
    parameter_arrays,
    positive_parameter,
)


_SQUARE_CENTIMETERS_PER_SQUARE_METER = 1.0e4

# Smallest admissible growth texture factor 1 - 3 f_z of the axial direction.
_MINIMUM_AXIAL_GROWTH_TEXTURE_FACTOR = 1.0e-6

# FRAPCON-4.0 CREPR listing (PNNL-19417 Appendix A.2.8): gas constant
# [J/(mol K)], irradiation-creep temperature clamp [K], primary creep onset
# [h], and the factor applied to the summed rate before it is returned.
_CREEP_GAS_CONSTANT = 8.314
_IRRADIATION_CREEP_MINIMUM_TEMPERATURE = 570.0
_IRRADIATION_CREEP_MAXIMUM_TEMPERATURE = 625.0
_PRIMARY_CREEP_ONSET_HOURS = 0.1
_CREEP_EFFECTIVE_RATE_FACTOR = 1.1547
# Limback-Andersson primary creep (FRAPCON-4.0 Eqs. (2.90)-(2.92)).
_PRIMARY_SATURATION_COEFFICIENT = 0.0216
_PRIMARY_SATURATION_RATE_EXPONENT = 0.109
_PRIMARY_SATURATION_TANH_SCALE = 3.55e4
_PRIMARY_SATURATION_TANH_EXPONENT = -2.05
_PRIMARY_RATE_CONSTANT = 52.0
# FRAPCON-4.0 Section 2.4.1.1: a change of the signed effective stress larger
# than 5 MPa [Pa] starts a new primary-creep episode.
_PRIMARY_STRESS_CHANGE_THRESHOLD = 5.0e6
_PRIMARY_CREEP_HISTORY_ID = "frapcon-4.0:primary-creep-stress-change-5MPa"

# FRAPCON-4.0 Section 2.4.1.3 / PNNL-17700 Eq. (2): reference strain rate [1/s].
_REFERENCE_STRAIN_RATE = 1.0e-3
# FRAPCON-4.0 Section 2.4.1.3 / PNNL-17700 Sections 2.2-2.4: CKMN regime
# boundaries. Temperatures [K]: end of the low-temperature K(T) and m(T) fits,
# start of the beta K(T) fit, the n(T) plateau and alpha limits, start of the
# beta n(T) plateau, and end of the m(T) transition. The alpha K(T) fit ends at
# ``ZIRCALOY_ALPHA_PHASE_MAXIMUM_TEMPERATURE``.
_CKMN_LOW_TEMPERATURE_LIMIT = 750.0
_CKMN_STRENGTH_BETA_TEMPERATURE = 1255.0
_CKMN_HARDENING_LOW_TEMPERATURE_LIMIT = 419.4
_CKMN_HARDENING_ALPHA_TEMPERATURE_LIMIT = 1099.0772
_CKMN_HARDENING_BETA_TEMPERATURE = 1600.0
_CKMN_RATE_EXPONENT_TRANSITION_END = 800.0
# Fast fluences [n/m^2]: end of the low-fluence K(Phi) and n(Phi) fits, end of
# the intermediate fits, and saturation of n(Phi).
_CKMN_LOW_FLUENCE_LIMIT = 0.1e25
_CKMN_INTERMEDIATE_FLUENCE_LIMIT = 2.0e25
_CKMN_HARDENING_SATURATION_FLUENCE = 7.5e25
# Lower bound [Pa] of the stress scale that conditions the scaled unknown of the
# plastic return root.
_PLASTIC_ROOT_STRESS_SCALE_FLOOR = 1.0


def _stress_tensor(value: ArrayLike, /) -> Array:
    tensor = float64_array(value)
    if tensor.ndim < 2 or tensor.shape[-2:] != (3, 3):
        raise ValueError("Deviatoric stress must have trailing shape (3, 3).")
    return tensor


@final
class ZircaloyCreepState(StrictModule):
    """Accepted primary-creep history; stress axes are radial, hoop, axial."""

    creep_time: Array
    reference_deviatoric_stress: Array
    primary_deviatoric_stress: Array
    primary_equivalent_stress: Array
    accumulated_equivalent_strain: Array
    history_id: str = eqx.field(static=True, default=_PRIMARY_CREEP_HISTORY_ID)

    @classmethod
    def initialize(cls, deviatoric_stress: ArrayLike, /) -> ZircaloyCreepState:
        stress = _stress_tensor(deviatoric_stress)
        scalar = jnp.zeros(stress.shape[:-2], dtype=stress.dtype)
        return cls(
            creep_time=scalar,
            reference_deviatoric_stress=jnp.zeros_like(stress),
            primary_deviatoric_stress=jnp.zeros_like(stress),
            primary_equivalent_stress=scalar,
            accumulated_equivalent_strain=scalar,
        )


@final
class ZircaloyCreepIncrement(StrictModule):
    strain_increment: Array
    equivalent_strain_increment: Array
    state: ZircaloyCreepState


@final
class ZircaloyPlasticityResult(StrictModule):
    deviatoric_stress: Array
    plastic_strain_increment: Array
    equivalent_plastic_strain: Array
    converged: Array
    residual: Array


def _primary_saturation_strain(steady_rate: Array, /) -> Array:
    """Limback-Andersson saturated primary strain of a positive steady rate [1/h]."""
    return (
        _PRIMARY_SATURATION_COEFFICIENT
        * steady_rate**_PRIMARY_SATURATION_RATE_EXPONENT
        * (2.0 - jnp.tanh(_PRIMARY_SATURATION_TANH_SCALE * steady_rate))
        ** _PRIMARY_SATURATION_TANH_EXPONENT
    )


def _hoop_signed(equivalent: Array, deviatoric_stress: Array, /) -> Array:
    """Equivalent stress carrying the sign of the hoop stress component."""
    return jnp.where(deviatoric_stress[..., 1, 1] < 0.0, -equivalent, equivalent)


def _flow_direction(deviatoric_stress: Array, equivalent: Array, /) -> Array:
    """``s / q`` of a deviatoric stress, the zero tensor where ``q = 0``."""
    return (
        deviatoric_stress / jnp.where(equivalent > 0.0, equivalent, 1.0)[..., None, None]
    )


@final
class ZircaloyCreep(StrictModule):
    """Thermal, irradiation, and primary creep of SRA Zircaloy (FRAPCON-4.0 CREPR).

    The Limback-Andersson model (ASTM STP 1295, 1996, pp. 448-468) as modified
    by PNNL for effective stress, FRAPCON-4.0 Eqs. (2.87)-(2.92) and Table 2.3,
    evaluated exactly as the CREPR listing (PNNL-19417 Rev. 2 Appendix A.2.8)
    in its source units (MPa, K, n/(m^2 s), n/cm^2, hours):

    ``rate_th = A (E/T) sinh(a_i s / E)^n exp(-Q / (R T))``,
    ``E = 1.148e5 - 59.9 T`` [MPa],
    ``a_i = 650 (1 - 0.56 (1 - exp(-1.4e-27 Phi^1.3)))``, ``Phi`` [n/cm^2],
    ``rate_irr = C0 phi^0.85 s f(T)``, ``f(T) = -7.0237 + 0.0136 T`` with ``T``
    clamped to [570, 625] K,
    ``rate_s = rate_th + rate_irr``, ``eps_ps = 0.0216 rate_s^0.109 (2 -
    tanh(3.55e4 rate_s))^-2.05``, and for ``t >= 0.1 h`` the primary rate
    ``0.5 eps_ps 52 rate_s^0.5 t^-0.5 exp(-52 (rate_s t)^0.5)``. The summed
    rate is multiplied by 1.1547 and returned in 1/s as the equivalent creep
    strain rate of the Prandtl-Reuss creep flow rule (FRAPCON-4.0 Eq. (2.84)).

    ``creep_time`` [s] is the primary-creep clock. ``equivalent_creep_rate`` is
    the constant-stress CREPR correlation; ``advance`` is the history-aware
    tensor update that resets the clock and the primary-creep stress on a
    significant stress change, including the FRAPCON-4.0 Section 2.4.1.1
    reversal rule.
    """

    thermal_coefficient: Array
    modulus_intercept: Array
    modulus_slope: Array
    stress_sensitivity: Array
    sensitivity_reduction: Array
    sensitivity_fluence_coefficient: Array
    sensitivity_fluence_exponent: Array
    thermal_stress_exponent: Array
    activation_energy: Array
    irradiation_coefficient: Array
    flux_exponent: Array
    irradiation_stress_exponent: Array
    irradiation_temperature_intercept: Array
    irradiation_temperature_slope: Array
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
        thermal_coefficient: ArrayLike = 1.08e9,
        modulus_intercept: ArrayLike = 1.148e5,
        modulus_slope: ArrayLike = 59.9,
        stress_sensitivity: ArrayLike = 650.0,
        sensitivity_reduction: ArrayLike = 0.56,
        sensitivity_fluence_coefficient: ArrayLike = 1.4e-27,
        sensitivity_fluence_exponent: ArrayLike = 1.3,
        thermal_stress_exponent: ArrayLike = 2.0,
        activation_energy: ArrayLike = 201000.0,
        irradiation_coefficient: ArrayLike = 4.0985e-24,
        flux_exponent: ArrayLike = 0.85,
        irradiation_stress_exponent: ArrayLike = 1.0,
        irradiation_temperature_intercept: ArrayLike = -7.0237,
        irradiation_temperature_slope: ArrayLike = 0.0136,
    ) -> None:
        parameters = {
            "thermal_coefficient": positive_parameter(
                thermal_coefficient, "thermal_coefficient"
            ),
            "modulus_intercept": positive_parameter(
                modulus_intercept, "modulus_intercept"
            ),
            "modulus_slope": finite_parameter(modulus_slope, "modulus_slope"),
            "stress_sensitivity": positive_parameter(
                stress_sensitivity, "stress_sensitivity"
            ),
            "sensitivity_reduction": finite_parameter(
                sensitivity_reduction, "sensitivity_reduction"
            ),
            "sensitivity_fluence_coefficient": positive_parameter(
                sensitivity_fluence_coefficient, "sensitivity_fluence_coefficient"
            ),
            "sensitivity_fluence_exponent": positive_parameter(
                sensitivity_fluence_exponent, "sensitivity_fluence_exponent"
            ),
            "thermal_stress_exponent": positive_parameter(
                thermal_stress_exponent, "thermal_stress_exponent"
            ),
            "activation_energy": positive_parameter(
                activation_energy, "activation_energy"
            ),
            "irradiation_coefficient": positive_parameter(
                irradiation_coefficient, "irradiation_coefficient"
            ),
            "flux_exponent": positive_parameter(flux_exponent, "flux_exponent"),
            "irradiation_stress_exponent": positive_parameter(
                irradiation_stress_exponent, "irradiation_stress_exponent"
            ),
            "irradiation_temperature_intercept": finite_parameter(
                irradiation_temperature_intercept, "irradiation_temperature_intercept"
            ),
            "irradiation_temperature_slope": finite_parameter(
                irradiation_temperature_slope, "irradiation_temperature_slope"
            ),
        }
        arrays = parameter_arrays(parameters)
        self.thermal_coefficient = arrays["thermal_coefficient"]
        self.modulus_intercept = arrays["modulus_intercept"]
        self.modulus_slope = arrays["modulus_slope"]
        self.stress_sensitivity = arrays["stress_sensitivity"]
        self.sensitivity_reduction = arrays["sensitivity_reduction"]
        self.sensitivity_fluence_coefficient = arrays["sensitivity_fluence_coefficient"]
        self.sensitivity_fluence_exponent = arrays["sensitivity_fluence_exponent"]
        self.thermal_stress_exponent = arrays["thermal_stress_exponent"]
        self.activation_energy = arrays["activation_energy"]
        self.irradiation_coefficient = arrays["irradiation_coefficient"]
        self.flux_exponent = arrays["flux_exponent"]
        self.irradiation_stress_exponent = arrays["irradiation_stress_exponent"]
        self.irradiation_temperature_intercept = arrays[
            "irradiation_temperature_intercept"
        ]
        self.irradiation_temperature_slope = arrays["irradiation_temperature_slope"]
        self.source_id = "frapcon-4.0-pnnl-19417-rev2:CREPR"
        # PNNL-19417 Section 3.8.2: 570-625 K, 40-130 MPa effective stress,
        # fast flux 1e17-2e18 n/(m^2 s).
        self.validity = PropertyValidityRange(
            {
                "temperature": (570.0, 625.0, "K"),
                "equivalent_stress": (40.0e6, 130.0e6, "Pa"),
                "fast_flux": (1.0e17, 2.0e18, "n/(m^2 s)"),
            },
            source_id=self.source_id,
        )
        self.law_id = canonical_fingerprint(
            {
                "kind": "zircaloy-sra-creep",
                "alloy": self.alloy,
                "heat_treatment": self.heat_treatment,
                "history": _PRIMARY_CREEP_HISTORY_ID,
                "parameters": parameters,
            }
        )

    def equivalent_creep_rate(
        self,
        equivalent_stress: ArrayLike,
        temperature: ArrayLike,
        fast_flux: ArrayLike,
        fast_fluence: ArrayLike,
        creep_time: ArrayLike,
        /,
    ) -> Array:
        """Equivalent creep strain rate [1/s].

        ``equivalent_stress`` [Pa] is the von Mises stress, ``fast_flux``
        [n/(m^2 s)], ``fast_fluence`` [n/m^2], and ``creep_time`` [s] the
        primary-creep clock.
        """

        stress = jnp.abs(float64_array(equivalent_stress)) / PASCAL_PER_MEGAPASCAL
        t = float64_array(temperature)
        flux = float64_array(fast_flux)
        fluence = float64_array(fast_fluence) / _SQUARE_CENTIMETERS_PER_SQUARE_METER
        hours = float64_array(creep_time) / SECONDS_PER_HOUR
        modulus = self.modulus_intercept - self.modulus_slope * t
        sensitivity = self.stress_sensitivity * (
            1.0
            - self.sensitivity_reduction
            * (
                1.0
                - jnp.exp(
                    -self.sensitivity_fluence_coefficient
                    * zero_safe_power(fluence, self.sensitivity_fluence_exponent)
                )
            )
        )
        thermal = (
            self.thermal_coefficient
            * (modulus / t)
            * jnp.sinh(sensitivity * stress / modulus) ** self.thermal_stress_exponent
            * jnp.exp(-self.activation_energy / (_CREEP_GAS_CONSTANT * t))
        )
        clamped = jnp.clip(
            t,
            _IRRADIATION_CREEP_MINIMUM_TEMPERATURE,
            _IRRADIATION_CREEP_MAXIMUM_TEMPERATURE,
        )
        irradiation = (
            self.irradiation_coefficient
            * zero_safe_power(flux, self.flux_exponent)
            * zero_safe_power(stress, self.irradiation_stress_exponent)
            * (
                self.irradiation_temperature_intercept
                + self.irradiation_temperature_slope * clamped
            )
        )
        steady = thermal + irradiation
        nonzero = steady > 0.0
        safe_steady = jnp.where(nonzero, steady, 1.0)
        saturation = _primary_saturation_strain(safe_steady)
        active = hours >= _PRIMARY_CREEP_ONSET_HOURS
        # The primary term is singular at t = 0; the onset gate keeps the
        # unselected branch finite so derivatives stay defined.
        elapsed = jnp.maximum(hours, _PRIMARY_CREEP_ONSET_HOURS)
        root_rate = jnp.sqrt(safe_steady)
        primary = (
            0.5
            * saturation
            * _PRIMARY_RATE_CONSTANT
            * root_rate
            / jnp.sqrt(elapsed)
            * jnp.exp(-_PRIMARY_RATE_CONSTANT * root_rate * jnp.sqrt(elapsed))
        )
        total = steady + jnp.where(active & nonzero, primary, 0.0)
        return _CREEP_EFFECTIVE_RATE_FACTOR * total / SECONDS_PER_HOUR

    def advance(
        self,
        deviatoric_stress: ArrayLike,
        temperature: ArrayLike,
        fast_flux: ArrayLike,
        fast_fluence: ArrayLike,
        state: ZircaloyCreepState,
        time_step: ArrayLike,
        /,
    ) -> ZircaloyCreepIncrement:
        """Candidate creep history at end-step stress, with exact frozen-rate integral.

        Stress [Pa] has radial/hoop/axial tensor axes; temperature [K], flux
        [n/(m² s)], fluence [n/m²], time step [s]. Primary creep restarts at
        each change of the hoop-signed effective stress larger than 5 MPa and
        follows the direction of that stress change. Secondary creep follows
        the current Prandtl-Reuss direction. The input history is not mutated.
        """
        if not isinstance(state, ZircaloyCreepState):
            raise TypeError("state must be a ZircaloyCreepState.")
        if state.history_id != _PRIMARY_CREEP_HISTORY_ID:
            raise ValueError(
                "Creep history identity does not match the stress-change law."
            )
        stress = _stress_tensor(deviatoric_stress)
        (
            equivalent,
            t,
            flux,
            fluence,
            dt,
            accepted_clock,
            accepted_primary,
            accumulated,
        ) = jnp.broadcast_arrays(
            von_mises_equivalent(stress),
            float64_array(temperature),
            float64_array(fast_flux),
            float64_array(fast_fluence),
            float64_array(time_step),
            state.creep_time,
            state.primary_equivalent_stress,
            state.accumulated_equivalent_strain,
        )
        tensor_shape = equivalent.shape + (3, 3)
        stress = jnp.broadcast_to(stress, tensor_shape)
        reference = jnp.broadcast_to(state.reference_deviatoric_stress, tensor_shape)
        accepted_primary_stress = jnp.broadcast_to(
            state.primary_deviatoric_stress, tensor_shape
        )

        # Episode reset: a significant hoop-signed stress change restarts the
        # primary clock and becomes the stress driving primary creep.
        signed_change = jnp.abs(
            _hoop_signed(equivalent, stress)
            - _hoop_signed(von_mises_equivalent(reference), reference)
        )
        changed = signed_change > _PRIMARY_STRESS_CHANGE_THRESHOLD
        clock = jnp.where(changed, 0.0, accepted_clock)
        primary_stress = jnp.where(
            changed[..., None, None], stress - reference, accepted_primary_stress
        )
        primary_equivalent = jnp.where(changed, signed_change, accepted_primary)

        # Rates frozen at the endpoint stress; the primary term is integrated
        # exactly over [clock, clock + dt] in hours.
        secondary = self.equivalent_creep_rate(equivalent, t, flux, fluence, 0.0) * dt
        primary_rate = self.equivalent_creep_rate(
            primary_equivalent, t, flux, fluence, 0.0
        ) * (SECONDS_PER_HOUR / _CREEP_EFFECTIVE_RATE_FACTOR)
        positive = primary_rate > 0.0
        safe_rate = jnp.where(positive, primary_rate, 1.0)
        saturation = _primary_saturation_strain(safe_rate)
        start = jnp.maximum(clock / SECONDS_PER_HOUR, _PRIMARY_CREEP_ONSET_HOURS)
        end = jnp.maximum((clock + dt) / SECONDS_PER_HOUR, _PRIMARY_CREEP_ONSET_HOURS)
        start_exponent = _PRIMARY_RATE_CONSTANT * jnp.sqrt(safe_rate * start)
        end_exponent = _PRIMARY_RATE_CONSTANT * jnp.sqrt(safe_rate * end)
        primary = jnp.where(
            positive,
            _CREEP_EFFECTIVE_RATE_FACTOR
            * saturation
            * jnp.exp(-start_exponent)
            * (-jnp.expm1(start_exponent - end_exponent)),
            0.0,
        )

        strain = 1.5 * (
            secondary[..., None, None] * _flow_direction(stress, equivalent)
            + primary[..., None, None]
            * _flow_direction(primary_stress, von_mises_equivalent(primary_stress))
        )
        increment = (2.0 / 3.0) * von_mises_equivalent(strain)
        candidate = ZircaloyCreepState(
            creep_time=clock + dt,
            reference_deviatoric_stress=jnp.where(
                changed[..., None, None], stress, reference
            ),
            primary_deviatoric_stress=primary_stress,
            primary_equivalent_stress=primary_equivalent,
            accumulated_equivalent_strain=accumulated + increment,
        )
        return ZircaloyCreepIncrement(
            strain_increment=strain,
            equivalent_strain_increment=increment,
            state=candidate,
        )


def _strength_temperature_term(t: Array, /) -> Array:
    # FRAPCON-4.0 Section 2.4.1.3 / PNNL-17700 Section 2.2: K(T) [Pa].
    low = 1.17628e9 + t * (4.54859e5 + t * (-3.28185e3 + t * 1.72752))
    alpha = 2.522488e6 * jnp.exp(2.8500027e6 / t**2)
    transition = 1.841376039e8 - 1.4345448e5 * t
    beta = 4.330e7 + t * (-6.685e4 + t * (3.7579e1 - t * 7.33e-3))
    return jnp.where(
        t < _CKMN_LOW_TEMPERATURE_LIMIT,
        low,
        jnp.where(
            t < ZIRCALOY_ALPHA_PHASE_MAXIMUM_TEMPERATURE,
            alpha,
            jnp.where(t < _CKMN_STRENGTH_BETA_TEMPERATURE, transition, beta),
        ),
    )


def _strength_fluence_term(fluence: Array, work: Array, t: Array, /) -> Array:
    # FRAPCON-4.0 Section 2.4.1.3: K(Phi) with the low-fluence factor f(CW, T).
    annealing = (
        2.25 * jnp.exp(-20.0 * work) * jnp.exp(jnp.minimum((t - 550.0) / 10.0, 0.0)) + 1.0
    )
    return jnp.where(
        fluence < _CKMN_LOW_FLUENCE_LIMIT,
        (-0.1464 + 1.464e-25 * fluence) * annealing,
        jnp.where(
            fluence < _CKMN_INTERMEDIATE_FLUENCE_LIMIT,
            2.928e-26 * fluence,
            0.53236 + 2.6618e-27 * fluence,
        ),
    )


def _hardening_temperature_term(t: Array, /) -> Array:
    # FRAPCON-4.0 Section 2.4.1.3 / PNNL-17700 Section 2.3: n(T).
    alpha = -9.490e-2 + t * (1.165e-3 + t * (-1.992e-6 + t * 9.588e-10))
    return jnp.where(
        t < _CKMN_HARDENING_LOW_TEMPERATURE_LIMIT,
        0.11405,
        jnp.where(
            t < _CKMN_HARDENING_ALPHA_TEMPERATURE_LIMIT,
            alpha,
            jnp.where(
                t < _CKMN_HARDENING_BETA_TEMPERATURE,
                -0.22655119 + 2.5e-4 * t,
                0.17344880,
            ),
        ),
    )


def _hardening_fluence_term(fluence: Array, /) -> Array:
    # FRAPCON-4.0 Section 2.4.1.3 / PNNL-17700 Section 2.3: n(Phi).
    return jnp.where(
        fluence < _CKMN_LOW_FLUENCE_LIMIT,
        1.321 + 0.48e-25 * fluence,
        jnp.where(
            fluence < _CKMN_INTERMEDIATE_FLUENCE_LIMIT,
            1.369 + 0.096e-25 * fluence,
            jnp.where(
                fluence < _CKMN_HARDENING_SATURATION_FLUENCE,
                1.5435 + 0.008727e-25 * fluence,
                1.608953,
            ),
        ),
    )


@final
class ZircaloyPlasticity(StrictModule):
    """Strain-hardening, rate-sensitive flow of Zircaloy-4 (FRAPCON-4.0 CKMN).

    The PNNL revision of the MATPRO equation of state (FRAPCON-4.0 Section
    2.4.1.3; PNNL-17700 Sections 2.2-2.4): ``K = K(T) (1 + 0.546 CW +
    K(Phi)) / K(Zry)``, ``n = n(T) n(Phi) / n(Zry)``, and ``m(T)``, with
    ``K(Zry) = n(Zry) = 1`` for Zircaloy-4. The piecewise branches are
    evaluated as published, including the small jumps of ``K(Phi)`` and
    ``n(Phi)`` at ``Phi = 1e24 n/m^2``.

    The flow stress is FRAPCON's subsequent yield stress for accumulated
    equivalent plastic strain ``eps_p`` (FRAPCON-4.0 Eqs. (2.134)-(2.135)):
    the root of ``s = K (s / E + eps_p)^n (rate / 1e-3)^m`` with ``E`` from
    ``ZircaloyElasticity`` (oxygen excess zero, as FRAPCON hardwires). At
    ``eps_p = 0`` it is the yield stress of PNNL-17700 Eq. (3). The residual is
    convex, so Newton from the upper bound ``max(2^(n/(1-n)) s_y, A (2
    eps_p)^n)`` converges monotonically; derivatives follow from the implicit
    function theorem. A non-positive rate returns the power-law limit zero.
    """

    cold_work_strengthening: Array
    alloy_strength_divisor: Array
    alloy_hardening_divisor: Array
    elasticity: ZircaloyElasticity
    root_plan: LocalRootPlan
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
        elasticity: ZircaloyElasticity | None = None,
        cold_work_strengthening: ArrayLike = 0.546,
        alloy_strength_divisor: ArrayLike = 1.0,
        alloy_hardening_divisor: ArrayLike = 1.0,
    ) -> None:
        modulus_law = ZircaloyElasticity() if elasticity is None else elasticity
        if not isinstance(modulus_law, ZircaloyElasticity):
            raise TypeError("elasticity must be a ZircaloyElasticity.")
        parameters = {
            "cold_work_strengthening": finite_parameter(
                cold_work_strengthening, "cold_work_strengthening"
            ),
            "alloy_strength_divisor": positive_parameter(
                alloy_strength_divisor, "alloy_strength_divisor"
            ),
            "alloy_hardening_divisor": positive_parameter(
                alloy_hardening_divisor, "alloy_hardening_divisor"
            ),
        }
        arrays = parameter_arrays(parameters)
        self.cold_work_strengthening = arrays["cold_work_strengthening"]
        self.alloy_strength_divisor = arrays["alloy_strength_divisor"]
        self.alloy_hardening_divisor = arrays["alloy_hardening_divisor"]
        self.elasticity = modulus_law
        self.root_plan = LocalRootPlan(
            maximum_steps=40, tolerance=1.0e-13, plan_id="zircaloy-subsequent-yield"
        )
        self.source_id = "frapcon-4.0-pnnl-19418-vol1-rev2:section-2.4.1.3;pnnl-17700"
        # PNNL-17700 Table 2 and Section 1 (PNNL and PROMETRA databases span
        # 2e-6 to 5 1/s); FRAPCON-4.0 Section 2.4.1.3 for cold work 0-0.75.
        self.validity = PropertyValidityRange(
            {
                "temperature": (293.0, 1273.0, "K"),
                "fast_fluence": (0.0, 1.4e26, "n/m^2"),
                "cold_work": (0.0, 0.75, "1"),
                "equivalent_plastic_strain_rate": (2.0e-6, 5.0, "1/s"),
            },
            source_id=self.source_id,
        )
        self.law_id = canonical_fingerprint(
            {
                "kind": "zircaloy-plasticity",
                "alloy": self.alloy,
                "heat_treatment": self.heat_treatment,
                "parameters": parameters,
                "elasticity": modulus_law.law_id,
                "root_plan": self.root_plan.plan_id,
            }
        )

    def strength_coefficient(
        self, temperature: ArrayLike, fast_fluence: ArrayLike, cold_work: ArrayLike, /
    ) -> Array:
        """Strength coefficient ``K`` [Pa]."""

        t, fluence, work = (
            float64_array(temperature),
            float64_array(fast_fluence),
            float64_array(cold_work),
        )
        return (
            _strength_temperature_term(t)
            * (
                1.0
                + self.cold_work_strengthening * work
                + _strength_fluence_term(fluence, work, t)
            )
            / self.alloy_strength_divisor
        )

    def strain_hardening_exponent(
        self, temperature: ArrayLike, fast_fluence: ArrayLike, /
    ) -> Array:
        """Strain hardening exponent ``n``."""

        t, fluence = float64_array(temperature), float64_array(fast_fluence)
        return (
            _hardening_temperature_term(t)
            * _hardening_fluence_term(fluence)
            / self.alloy_hardening_divisor
        )

    def strain_rate_exponent(self, temperature: ArrayLike, /) -> Array:
        """Strain rate exponent ``m`` (FRAPCON-3.5/4.0 Section 2.4.1.3)."""

        t = float64_array(temperature)
        return jnp.where(
            t < _CKMN_LOW_TEMPERATURE_LIMIT,
            0.015,
            jnp.where(
                t < _CKMN_RATE_EXPONENT_TRANSITION_END,
                7.458e-4 * t - 0.544338,
                3.24124e-4 * t - 0.20701,
            ),
        )

    def _rate_amplitude(
        self, loading_rate: Array, t: Array, fluence: Array, work: Array, /
    ) -> Array:
        """``K (rate / 1e-3)^m`` [Pa]; a non-positive rate uses the reference rate."""

        return self.strength_coefficient(t, fluence, work) * (
            jnp.where(loading_rate > 0.0, loading_rate, _REFERENCE_STRAIN_RATE)
            / _REFERENCE_STRAIN_RATE
        ) ** self.strain_rate_exponent(t)

    def flow_stress(
        self,
        equivalent_plastic_strain: ArrayLike,
        equivalent_plastic_strain_rate: ArrayLike,
        temperature: ArrayLike,
        fast_fluence: ArrayLike,
        cold_work: ArrayLike,
        /,
    ) -> Array:
        """Subsequent yield stress [Pa] for equivalent plastic strain and rate [1/s]."""

        strain, rate, t, fluence, work = jnp.broadcast_arrays(
            float64_array(equivalent_plastic_strain),
            float64_array(equivalent_plastic_strain_rate),
            float64_array(temperature),
            float64_array(fast_fluence),
            float64_array(cold_work),
        )
        hardening = self.strain_hardening_exponent(t, fluence)
        modulus = self.elasticity.youngs_modulus(t, fluence, work, jnp.zeros_like(t))
        loading = rate > 0.0
        amplitude = self._rate_amplitude(rate, t, fluence, work)
        initial_yield = (amplitude / modulus**hardening) ** (1.0 / (1.0 - hardening))
        bound = jnp.maximum(
            2.0 ** (hardening / (1.0 - hardening)) * initial_yield,
            amplitude
            * (2.0 * jnp.where(strain > 0.0, strain, 1.0)) ** hardening
            * jnp.where(strain > 0.0, 1.0, 0.0),
        )

        def subsequent_yield(
            site_amplitude: Array,
            site_modulus: Array,
            site_hardening: Array,
            site_strain: Array,
            site_bound: Array,
        ) -> Array:
            # Unknown scaled by the upper bound, so Newton starts at one.
            def residual(scaled: Array, /) -> Array:
                return (
                    scaled
                    - site_amplitude
                    * (scaled * site_bound / site_modulus + site_strain) ** site_hardening
                    / site_bound
                )

            return site_bound * self.root_plan.solve(residual, jnp.ones_like(site_bound))

        stress = jax.vmap(subsequent_yield)(
            amplitude.reshape(-1),
            modulus.reshape(-1),
            hardening.reshape(-1),
            strain.reshape(-1),
            bound.reshape(-1),
        ).reshape(strain.shape)
        return jnp.where(loading, stress, 0.0)

    def return_map(
        self,
        trial_deviatoric_stress: ArrayLike,
        equivalent_plastic_strain: ArrayLike,
        equivalent_strain_rate: ArrayLike,
        temperature: ArrayLike,
        fast_fluence: ArrayLike,
        cold_work: ArrayLike,
        time_step: ArrayLike,
        /,
        *,
        moduli: ZircaloyElasticModuli | None = None,
    ) -> ZircaloyPlasticityResult:
        """Isotropic J2 radial return of trial deviatoric stress [Pa].

        The effective *loading* strain rate [1/s] enters the PNNL power law.
        It is not the plastic strain increment divided by the time step.
        Accumulated plastic strain is dimensionless; ``time_step`` [s]
        distinguishes a static zero-duration evaluation from a loading step.
        Zero loading rate retains the elastic trial stress (an unloaded hold).
        ``moduli`` supplies the Young's and shear moduli; by default they are
        ``elasticity`` at zero excess oxygen. Root diagnostics are returned for
        candidate acceptance.
        """
        if moduli is None:
            moduli_temperature = float64_array(temperature)
            moduli = self.elasticity.moduli(
                moduli_temperature,
                float64_array(fast_fluence),
                float64_array(cold_work),
                jnp.zeros_like(moduli_temperature),
            )
        trial = _stress_tensor(trial_deviatoric_stress)
        (
            trial_equivalent,
            plastic_strain,
            loading_rate,
            t,
            fluence,
            work,
            dt,
            youngs,
            shear,
        ) = jnp.broadcast_arrays(
            von_mises_equivalent(trial),
            float64_array(equivalent_plastic_strain),
            float64_array(equivalent_strain_rate),
            float64_array(temperature),
            float64_array(fast_fluence),
            float64_array(cold_work),
            float64_array(time_step),
            float64_array(moduli.youngs),
            float64_array(moduli.shear),
        )
        shape = trial_equivalent.shape
        trial = jnp.broadcast_to(trial, shape + (3, 3))
        hardening = self.strain_hardening_exponent(t, fluence)
        amplitude = self._rate_amplitude(loading_rate, t, fluence, work)
        scale = jnp.maximum(trial_equivalent, _PLASTIC_ROOT_STRESS_SCALE_FLOOR)
        trial_strain = trial_equivalent / youngs + plastic_strain
        safe_trial_strain = jnp.where(trial_strain > 0.0, trial_strain, 1.0)
        yielded = (
            (trial_equivalent > amplitude * safe_trial_strain**hardening)
            & (loading_rate > 0.0)
            & (dt > 0.0)
        )

        def solve_site(
            site_trial_equivalent: Array,
            site_plastic_strain: Array,
            site_youngs: Array,
            site_shear: Array,
            site_hardening: Array,
            site_amplitude: Array,
            site_scale: Array,
            site_yielded: Array,
        ) -> tuple[Array, Array, Array]:
            # Unknown: returned equivalent stress over site_scale. Elastic sites
            # solve the identity residual for the trial stress.
            def residual(scaled: Array, /) -> Array:
                stress = scaled * site_scale
                increment = (site_trial_equivalent - stress) / (3.0 * site_shear)
                total = stress / site_youngs + site_plastic_strain + increment
                safe_total = jnp.where(site_yielded, total, 1.0)
                plastic = (
                    scaled - site_amplitude * safe_total**site_hardening / site_scale
                )
                return jnp.where(
                    site_yielded, plastic, scaled - site_trial_equivalent / site_scale
                )

            root, diagnostics = self.root_plan.solve_with_diagnostics(
                residual, site_trial_equivalent / site_scale
            )
            return root * site_scale, diagnostics.converged, diagnostics.residual

        mapped, converged, residual = jax.vmap(solve_site)(
            trial_equivalent.reshape(-1),
            plastic_strain.reshape(-1),
            youngs.reshape(-1),
            shear.reshape(-1),
            hardening.reshape(-1),
            amplitude.reshape(-1),
            scale.reshape(-1),
            yielded.reshape(-1),
        )
        mapped = mapped.reshape(shape)
        converged = (
            converged.reshape(shape)
            & (dt >= 0.0)
            & (plastic_strain >= 0.0)
            & (loading_rate >= 0.0)
        )
        residual = residual.reshape(shape)
        increment = jnp.where(yielded, (trial_equivalent - mapped) / (3.0 * shear), 0.0)
        converged = (
            converged
            & (increment >= 0.0)
            & (mapped >= 0.0)
            & (mapped <= trial_equivalent)
        )
        plastic_tensor = (1.5 * increment)[..., None, None] * _flow_direction(
            trial, trial_equivalent
        )
        stress = trial - (2.0 * shear)[..., None, None] * plastic_tensor
        return ZircaloyPlasticityResult(
            deviatoric_stress=stress,
            plastic_strain_increment=plastic_tensor,
            equivalent_plastic_strain=plastic_strain + increment,
            converged=converged,
            residual=residual,
        )


@final
class ZircaloyIrradiationGrowth(StrictModule):
    """Irradiation growth strains of SRA Zircaloy-4 in a PWR.

    Axial growth is Franklin's model as implemented in FRAPCON-4.0 CAGROW
    (PNNL-19417 Rev. 2 Eq. (3.44) and Appendix A.2.7): ``eps_z = 2.18e-21
    Phi^0.845`` with ``Phi`` the fast fluence in n/cm^2 (E > 1 MeV); FRAPCON
    has no temperature or texture dependence for this alloy. The hoop and
    radial components follow MATPRO's volume-conserving texture rule, growth
    along direction ``i`` proportional to ``1 - 3 f_i`` (NUREG/CR-6150 Rev. 2
    Vol. 4, Eq. (4-104)), scaled so the axial component equals FRAPCON's.
    """

    coefficient: Array
    fluence_exponent: Array
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
        coefficient: ArrayLike = 2.18e-21,
        fluence_exponent: ArrayLike = 0.845,
    ) -> None:
        texture_law = ZircaloyTexture() if texture is None else texture
        if not isinstance(texture_law, ZircaloyTexture):
            raise TypeError("texture must be a ZircaloyTexture.")
        if (
            not 1.0 - 3.0 * float(texture_law.axial)
            > _MINIMUM_AXIAL_GROWTH_TEXTURE_FACTOR
        ):
            raise ValueError(
                "Axial growth requires an axial Kearns factor below one third."
            )
        parameters = {
            "coefficient": positive_parameter(coefficient, "coefficient"),
            "fluence_exponent": positive_parameter(fluence_exponent, "fluence_exponent"),
        }
        arrays = parameter_arrays(parameters)
        self.coefficient = arrays["coefficient"]
        self.fluence_exponent = arrays["fluence_exponent"]
        self.texture = texture_law
        self.source_id = (
            "frapcon-4.0-pnnl-19417-rev2:CAGROW-eq-3.44;"
            "matpro-nureg-cr-6150-rev2-vol4:eq-4-104"
        )
        # PNNL-19417 Section 3.7.2: up to 12e25 n/m^2 (65 GWd/MTU).
        self.validity = PropertyValidityRange(
            {"fast_fluence": (0.0, 1.2e26, "n/m^2")}, source_id=self.source_id
        )
        self.law_id = canonical_fingerprint(
            {
                "kind": "zircaloy-irradiation-growth",
                "alloy": self.alloy,
                "heat_treatment": self.heat_treatment,
                "parameters": parameters,
                "texture": texture_law.texture_id,
            }
        )

    def principal_strains(self, fast_fluence: ArrayLike, /) -> CladdingPrincipalStrains:
        """Growth strains [m/m] at ``fast_fluence`` [n/m^2]."""

        fluence = float64_array(fast_fluence) / _SQUARE_CENTIMETERS_PER_SQUARE_METER
        axial = self.coefficient * zero_safe_power(fluence, self.fluence_exponent)
        per_texture = axial / (1.0 - 3.0 * self.texture.axial)
        return CladdingPrincipalStrains(
            radial=per_texture * (1.0 - 3.0 * self.texture.radial),
            hoop=per_texture * (1.0 - 3.0 * self.texture.hoop),
            axial=axial,
        )


__all__ = [
    "ZircaloyCreep",
    "ZircaloyCreepIncrement",
    "ZircaloyCreepState",
    "ZircaloyIrradiationGrowth",
    "ZircaloyPlasticity",
    "ZircaloyPlasticityResult",
]
