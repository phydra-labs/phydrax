#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Thermal and irradiation creep of sintered UO2 fuel pellets.

``UO2Creep`` returns an equivalent (von Mises) creep strain rate.

Primary source: Hagrman, Reymann (eds.), "MATPRO-Version 11", NUREG/CR-0497
(TREE-1280), 1979, Section A-6 (FCREEP) and its subcode listing, Table A-6.1;
the same model as NUREG/CR-6150 Vol. 4 Section 2.7.
"""

from __future__ import annotations

import math
from typing import final

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...materials import PropertyValidityRange
from ._material_math import zero_safe_log_power
from ._material_parameters import finite_parameter, float64_array, positive_parameter


_MICROMETER_PER_METER = 1.0e6
_PERCENT = 100.0
# FCREEP (NUREG/CR-0497 Table A-6.1) clamps O/M below 2 to 2 and evaluates the
# stoichiometry factor 1 / (exp(-20 / log10(O/M - 1.99999) - 8) + 1); the
# correlation is defined for O/M < 2.99999.
_FCREEP_STOICHIOMETRY_GUARD = 1.99999
_FCREEP_STOICHIOMETRIC_RATIO = 2.0
_FCREEP_STOICHIOMETRY_SCALE = 20.0
_FCREEP_STOICHIOMETRY_SHIFT = 8.0


@final
class FuelCreepState(StrictModule):
    """FCREEP history at an accepted or candidate material point."""

    previous_equivalent_stress: Array
    time_since_stress_increase: Array


@final
class FuelCreepUpdate(StrictModule):
    """Uncommitted backward-Euler FCREEP response with candidate history."""

    equivalent_creep_rate: Array
    equivalent_creep_strain_increment: Array
    state: FuelCreepState


@final
class UO2Creep(StrictModule):
    """Steady-state thermal and irradiation creep of UO2 (MATPRO FCREEP).

    MATPRO-11 (NUREG/CR-0497) Eq. (A-6.4) as coded in the FCREEP listing,
    Table A-6.1, with activation energies written as temperatures ``Q/R``
    (the listing evaluates ``exp(-Q/T)``)::

        eps = (A1 + A2 F) s_l exp(-Q1/T) / ((A3 + D) G^2)
              + [s > s_t] A4 s^4.5 exp(-Q2/T) / (A6 + D)
              + A7 F s exp(-Q3/T)

    ``s`` the equivalent stress [Pa], ``s_l = min(s, s_t)`` with the transition
    stress ``s_t = 1.6547e7 / G^0.5714`` (Eq. (A-6.1)), ``G`` the grain size in
    micrometers, ``D`` the density in percent of theoretical, ``F`` the fission
    rate density [fissions/(m^3 s)] and ``T`` [K]. Below ``s_t`` the listing
    uses the applied stress in the linear term and no power-law term; above,
    the transition stress in the linear term and the applied stress in the
    power-law term, so the rate jumps at ``s_t`` exactly as in the source. The
    activation temperatures depend on the O/M ratio ``x`` through
    ``f(x) = 1 / (exp(-20 / log10(x - 1.99999) - 8) + 1)`` with ``x`` clamped
    below at 2 (listing), ``Q1 = 9000 f + 36294.4`` and
    ``Q2 = 10000 f + 56431.8``, i.e. ``17884.8 f + 72124.23`` and
    ``19872 f + 112142.4`` cal/mol (Eqs. (A-6.2), (A-6.18)); ``Q3 = 2617 K``.
    ``equivalent_creep_rate`` gives the steady rate; ``update`` includes the
    primary factor ``1 + 2.5 exp(-1.2e-6 age)`` and the age-reset rule of the
    Table A-6.1 listing (candidate stress greater than previous-step stress).
    The report equation prints 1.4e-6 instead of the listing's 1.2e-6; the
    implementation follows the executable source consistently.
    """

    diffusional_coefficient: Array
    fission_enhancement_coefficient: Array
    diffusional_density_offset: Array
    dislocation_coefficient: Array
    dislocation_density_offset: Array
    dislocation_stress_exponent: Array
    irradiation_coefficient: Array
    diffusional_activation_temperature: Array
    diffusional_stoichiometry_temperature: Array
    dislocation_activation_temperature: Array
    dislocation_stoichiometry_temperature: Array
    irradiation_activation_temperature: Array
    transition_stress_coefficient: Array
    transition_grain_exponent: Array
    primary_amplitude: Array
    primary_decay_rate: Array
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        diffusional_coefficient: ArrayLike = 0.3919,
        fission_enhancement_coefficient: ArrayLike = 1.3100e-19,
        diffusional_density_offset: ArrayLike = -87.7,
        dislocation_coefficient: ArrayLike = 2.0391e-25,
        dislocation_density_offset: ArrayLike = -90.5,
        dislocation_stress_exponent: ArrayLike = 4.5,
        irradiation_coefficient: ArrayLike = 3.72264e-35,
        diffusional_activation_temperature: ArrayLike = 36294.4,
        diffusional_stoichiometry_temperature: ArrayLike = 9000.0,
        dislocation_activation_temperature: ArrayLike = 56431.8,
        dislocation_stoichiometry_temperature: ArrayLike = 10000.0,
        irradiation_activation_temperature: ArrayLike = 2617.0,
        transition_stress_coefficient: ArrayLike = 1.6547e7,
        transition_grain_exponent: ArrayLike = 0.5714,
        primary_amplitude: ArrayLike = 2.5,
        primary_decay_rate: ArrayLike = 1.2e-6,
    ) -> None:
        parameters = {
            "diffusional_coefficient": finite_parameter(
                diffusional_coefficient, "diffusional_coefficient"
            ),
            "fission_enhancement_coefficient": finite_parameter(
                fission_enhancement_coefficient, "fission_enhancement_coefficient"
            ),
            "diffusional_density_offset": finite_parameter(
                diffusional_density_offset, "diffusional_density_offset"
            ),
            "dislocation_coefficient": finite_parameter(
                dislocation_coefficient, "dislocation_coefficient"
            ),
            "dislocation_density_offset": finite_parameter(
                dislocation_density_offset, "dislocation_density_offset"
            ),
            "dislocation_stress_exponent": positive_parameter(
                dislocation_stress_exponent, "dislocation_stress_exponent"
            ),
            "irradiation_coefficient": finite_parameter(
                irradiation_coefficient, "irradiation_coefficient"
            ),
            "diffusional_activation_temperature": finite_parameter(
                diffusional_activation_temperature, "diffusional_activation_temperature"
            ),
            "diffusional_stoichiometry_temperature": finite_parameter(
                diffusional_stoichiometry_temperature,
                "diffusional_stoichiometry_temperature",
            ),
            "dislocation_activation_temperature": finite_parameter(
                dislocation_activation_temperature, "dislocation_activation_temperature"
            ),
            "dislocation_stoichiometry_temperature": finite_parameter(
                dislocation_stoichiometry_temperature,
                "dislocation_stoichiometry_temperature",
            ),
            "irradiation_activation_temperature": finite_parameter(
                irradiation_activation_temperature, "irradiation_activation_temperature"
            ),
            "transition_stress_coefficient": positive_parameter(
                transition_stress_coefficient, "transition_stress_coefficient"
            ),
            "transition_grain_exponent": finite_parameter(
                transition_grain_exponent, "transition_grain_exponent"
            ),
            "primary_amplitude": finite_parameter(primary_amplitude, "primary_amplitude"),
            "primary_decay_rate": positive_parameter(
                primary_decay_rate, "primary_decay_rate"
            ),
        }
        self.diffusional_coefficient = jnp.asarray(parameters["diffusional_coefficient"])
        self.fission_enhancement_coefficient = jnp.asarray(
            parameters["fission_enhancement_coefficient"]
        )
        self.diffusional_density_offset = jnp.asarray(
            parameters["diffusional_density_offset"]
        )
        self.dislocation_coefficient = jnp.asarray(parameters["dislocation_coefficient"])
        self.dislocation_density_offset = jnp.asarray(
            parameters["dislocation_density_offset"]
        )
        self.dislocation_stress_exponent = jnp.asarray(
            parameters["dislocation_stress_exponent"]
        )
        self.irradiation_coefficient = jnp.asarray(parameters["irradiation_coefficient"])
        self.diffusional_activation_temperature = jnp.asarray(
            parameters["diffusional_activation_temperature"]
        )
        self.diffusional_stoichiometry_temperature = jnp.asarray(
            parameters["diffusional_stoichiometry_temperature"]
        )
        self.dislocation_activation_temperature = jnp.asarray(
            parameters["dislocation_activation_temperature"]
        )
        self.dislocation_stoichiometry_temperature = jnp.asarray(
            parameters["dislocation_stoichiometry_temperature"]
        )
        self.irradiation_activation_temperature = jnp.asarray(
            parameters["irradiation_activation_temperature"]
        )
        self.transition_stress_coefficient = jnp.asarray(
            parameters["transition_stress_coefficient"]
        )
        self.transition_grain_exponent = jnp.asarray(
            parameters["transition_grain_exponent"]
        )
        self.primary_amplitude = jnp.asarray(parameters["primary_amplitude"])
        self.primary_decay_rate = jnp.asarray(parameters["primary_decay_rate"])
        # MATPRO states no temperature, stress, fission-rate, or grain-size
        # envelope; the samples were "generally above 95% theoretically dense"
        # (NUREG/CR-6150 Vol. 4 Section 2.7) and the O/M form is defined on
        # [2, 2.99999) (Table A-6.1 clamps below 2).
        self.validity = PropertyValidityRange(
            {
                "density_fraction": (0.95, 1.0, "1"),
                "oxygen_to_metal_ratio": (
                    _FCREEP_STOICHIOMETRIC_RATIO,
                    math.nextafter(_FCREEP_STOICHIOMETRY_GUARD + 1.0, -math.inf),
                    "1",
                ),
            },
            source_id="nureg-cr-6150-vol4:2.7:nureg-cr-0497-table-a-6.1",
        )
        self.source_id = "nureg-cr-0497:eq-a-6.1-a-6.4-table-a-6.1"
        self.law_id = canonical_fingerprint(
            {
                "kind": "uo2-creep-matpro-fcreep",
                "parameters": parameters,
                "source": self.source_id,
            }
        )

    def transition_stress(self, grain_size: ArrayLike, /) -> Array:
        """Viscous-to-power-law transition stress [Pa] at grain size [m]."""

        grain_um = float64_array(grain_size) * _MICROMETER_PER_METER
        return (
            self.transition_stress_coefficient / grain_um**self.transition_grain_exponent
        )

    def equivalent_creep_rate(
        self,
        equivalent_stress: ArrayLike,
        temperature: ArrayLike,
        fission_rate_density: ArrayLike,
        density_fraction: ArrayLike,
        grain_size: ArrayLike,
        oxygen_to_metal_ratio: ArrayLike,
        /,
    ) -> Array:
        """Steady equivalent creep strain rate [1/s].

        ``equivalent_stress`` [Pa] is the non-negative von Mises stress,
        ``temperature`` [K], ``fission_rate_density`` [fissions/(m^3 s)],
        ``density_fraction`` the initial (as-fabricated) fraction of
        theoretical density, as ``FDENS`` in Table A-6.1, and ``grain_size``
        the current grain size [m].
        """

        stress, t, fission_rate, density, grain, ratio = jnp.broadcast_arrays(
            float64_array(equivalent_stress),
            float64_array(temperature),
            float64_array(fission_rate_density),
            float64_array(density_fraction),
            float64_array(grain_size),
            float64_array(oxygen_to_metal_ratio),
        )
        grain_um = grain * _MICROMETER_PER_METER
        density_percent = _PERCENT * density
        transition = self.transition_stress(grain)
        stoichiometry = 1.0 / (
            jnp.exp(
                -_FCREEP_STOICHIOMETRY_SCALE
                / jnp.log10(
                    jnp.maximum(ratio, _FCREEP_STOICHIOMETRIC_RATIO)
                    - _FCREEP_STOICHIOMETRY_GUARD
                )
                - _FCREEP_STOICHIOMETRY_SHIFT
            )
            + 1.0
        )
        diffusional_activation = (
            self.diffusional_stoichiometry_temperature * stoichiometry
            + self.diffusional_activation_temperature
        )
        dislocation_activation = (
            self.dislocation_stoichiometry_temperature * stoichiometry
            + self.dislocation_activation_temperature
        )
        diffusional = (
            (
                self.diffusional_coefficient
                + self.fission_enhancement_coefficient * fission_rate
            )
            * jnp.minimum(stress, transition)
            * jnp.exp(-diffusional_activation / t)
            / ((self.diffusional_density_offset + density_percent) * grain_um * grain_um)
        )
        dislocation = jnp.where(
            stress > transition,
            self.dislocation_coefficient
            * zero_safe_log_power(stress, self.dislocation_stress_exponent)
            * jnp.exp(-dislocation_activation / t)
            / (self.dislocation_density_offset + density_percent),
            0.0,
        )
        irradiation = (
            self.irradiation_coefficient
            * fission_rate
            * stress
            * jnp.exp(-self.irradiation_activation_temperature / t)
        )
        return diffusional + dislocation + irradiation

    def update(
        self,
        equivalent_stress: ArrayLike,
        temperature: ArrayLike,
        fission_rate_density: ArrayLike,
        density_fraction: ArrayLike,
        grain_size: ArrayLike,
        oxygen_to_metal_ratio: ArrayLike,
        accepted_state: FuelCreepState,
        time_step: ArrayLike,
        /,
    ) -> FuelCreepUpdate:
        """FCREEP candidate rate and backward-Euler increment, without commit.

        SI inputs follow ``equivalent_creep_rate``; ``time_step`` and the
        history age are seconds, historical stress is Pa. The listing's
        ``XTIME = TLAST`` on a stress increase resets the endpoint age to the
        current step duration. Otherwise the accepted age advances by it.
        """

        stress, previous_stress, age, dt = jnp.broadcast_arrays(
            float64_array(equivalent_stress),
            accepted_state.previous_equivalent_stress,
            accepted_state.time_since_stress_increase,
            float64_array(time_step),
        )
        candidate_age = jnp.where(stress > previous_stress, dt, age + dt)
        steady_rate = self.equivalent_creep_rate(
            stress,
            temperature,
            fission_rate_density,
            density_fraction,
            grain_size,
            oxygen_to_metal_ratio,
        )
        rate = steady_rate * (
            1.0
            + self.primary_amplitude * jnp.exp(-self.primary_decay_rate * candidate_age)
        )
        endpoint_stress, endpoint_age = jnp.broadcast_arrays(
            jnp.broadcast_to(stress, rate.shape), candidate_age
        )
        return FuelCreepUpdate(
            equivalent_creep_rate=rate,
            equivalent_creep_strain_increment=dt * rate,
            state=FuelCreepState(
                previous_equivalent_stress=endpoint_stress,
                time_since_stress_increase=endpoint_age,
            ),
        )


__all__ = [
    "FuelCreepState",
    "FuelCreepUpdate",
    "UO2Creep",
]
