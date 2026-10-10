#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Grain-face bubbles, grain growth and grain-boundary micro-cracking of UO2.

* Grain-face bubbles of Pastore et al. (Nucl. Eng. Des. 256 (2013) 75): Speight
  and Beere vacancy absorption, van der Waals gas, lenticular geometry, White
  coalescence and venting of the bubbles that exceed the saturation coverage.
* Ainscough et al. (J. Nucl. Mater. 49 (1973) 117) grain growth to a limiting
  size.
* Barani et al. (J. Nucl. Mater. 486 (2017) 96) micro-cracking burst release
  on temperature variations with burnup healing.
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
from ._constants import (
    BOLTZMANN_J_PER_K,
    MOLAR_GAS_CONSTANT_J_PER_MOL_K,
    SECONDS_PER_HOUR,
)
from ._gas_math import (
    host_parameters,
    positive_quadratic_root,
)


# Ainscough et al. (1973) rate constant is quoted in um^2/h.
_SQUARE_METER_PER_SECOND_PER_SQUARE_MICROMETER_PER_HOUR = 1.0e-12 / SECONDS_PER_HOUR
# Floor of the grain-face coverage inside the Speight-Beere sink strength,
# whose ``-ln(F) / 2`` term diverges for an empty face.
_MINIMUM_COVERAGE = 1.0e-12


@final
class GrainFaceBubbles(StrictModule):
    """Lenticular grain-face bubbles of Pastore et al. (2013).

    Parameters from INL/CON-13-28389 (Pastore et al., 2013) Table I: grain
    boundary vacancy diffusivity ``8.86e-6 exp(-5.75e-19 / (k T))`` m^2/s and
    diffusion layer ``3.5e-10`` m (White, 2004), surface energy ``0.5`` J/m^2,
    semi-dihedral angle 50 degrees, van der Waals volume ``8.5e-29`` m^3,
    vacancy volume ``4.09e-29`` m^3, initial bubble density ``2e13`` m^-2 and
    saturation coverage ``0.5`` of intact faces.
    """

    vacancy_diffusivity_prefactor: Array
    vacancy_activation_energy: Array
    boundary_layer_thickness: Array
    surface_energy: Array
    semidihedral_angle: Array
    van_der_waals_volume: Array
    vacancy_volume: Array
    initial_bubble_number_density: Array
    saturation_coverage: Array
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        vacancy_diffusivity_prefactor: ArrayLike = 8.86e-6,
        vacancy_activation_energy: ArrayLike = 5.75e-19,
        boundary_layer_thickness: ArrayLike = 3.5e-10,
        surface_energy: ArrayLike = 0.5,
        semidihedral_angle: ArrayLike = math.radians(50.0),
        van_der_waals_volume: ArrayLike = 8.5e-29,
        vacancy_volume: ArrayLike = 4.09e-29,
        initial_bubble_number_density: ArrayLike = 2.0e13,
        saturation_coverage: ArrayLike = 0.5,
    ) -> None:
        parameters = host_parameters(
            {
                "vacancy_diffusivity_prefactor": vacancy_diffusivity_prefactor,
                "vacancy_activation_energy": vacancy_activation_energy,
                "boundary_layer_thickness": boundary_layer_thickness,
                "surface_energy": surface_energy,
                "semidihedral_angle": semidihedral_angle,
                "van_der_waals_volume": van_der_waals_volume,
                "vacancy_volume": vacancy_volume,
                "initial_bubble_number_density": initial_bubble_number_density,
                "saturation_coverage": saturation_coverage,
            },
            positive=(
                "vacancy_diffusivity_prefactor",
                "boundary_layer_thickness",
                "semidihedral_angle",
                "vacancy_volume",
                "initial_bubble_number_density",
                "saturation_coverage",
            ),
        )
        if not parameters["semidihedral_angle"] < 0.5 * math.pi:
            raise ValueError("semidihedral_angle must lie in (0, pi/2).")
        if not parameters["saturation_coverage"] < 1.0:
            raise ValueError("saturation_coverage must lie in (0, 1).")
        self.vacancy_diffusivity_prefactor = jnp.asarray(
            parameters["vacancy_diffusivity_prefactor"]
        )
        self.vacancy_activation_energy = jnp.asarray(
            parameters["vacancy_activation_energy"]
        )
        self.boundary_layer_thickness = jnp.asarray(
            parameters["boundary_layer_thickness"]
        )
        self.surface_energy = jnp.asarray(parameters["surface_energy"])
        self.semidihedral_angle = jnp.asarray(parameters["semidihedral_angle"])
        self.van_der_waals_volume = jnp.asarray(parameters["van_der_waals_volume"])
        self.vacancy_volume = jnp.asarray(parameters["vacancy_volume"])
        self.initial_bubble_number_density = jnp.asarray(
            parameters["initial_bubble_number_density"]
        )
        self.saturation_coverage = jnp.asarray(parameters["saturation_coverage"])
        self.source_id = "pastore-2013-ned-256-75;inl-con-13-28389:eq5-14,table-i"
        self.law_id = canonical_fingerprint(
            {"kind": "grain-face-bubbles", "parameters": parameters}
        )

    def lenticular_factor(self) -> Array:
        """``phi(theta) = 1 - 1.5 cos(theta) + 0.5 cos(theta)^3``."""

        cosine = jnp.cos(self.semidihedral_angle)
        return 1.0 - 1.5 * cosine + 0.5 * cosine**3

    def radius_of_curvature(self, volume: ArrayLike, /) -> Array:
        """``R = (3 V / (4 pi phi))^(1/3)`` [m] of a bubble of volume ``V``."""

        return jnp.cbrt(
            3.0 * jnp.asarray(volume) / (4.0 * jnp.pi * self.lenticular_factor())
        )

    def projected_area(self, volume: ArrayLike, /) -> Array:
        """``A = pi (R sin(theta))^2`` [m^2]."""

        radius = self.radius_of_curvature(volume)
        return jnp.pi * (radius * jnp.sin(self.semidihedral_angle)) ** 2

    def vacancy_diffusivity(self, temperature: ArrayLike, /) -> Array:
        """Grain-boundary vacancy diffusivity [m^2/s] (White, 2004)."""

        return self.vacancy_diffusivity_prefactor * jnp.exp(
            -self.vacancy_activation_energy
            / (BOLTZMANN_J_PER_K * jnp.asarray(temperature))
        )


def speight_beere_sink_strength(
    faces: GrainFaceBubbles,
    bubble_number_density: Array,
    volume: Array,
    saturation_coverage: Array,
) -> Array:
    """Sink strength ``S(F) = -(3 - F)(1 - F)/4 - ln(F)/2`` of bubbles of ``volume``.

    The coverage ``F`` is bounded to ``[_MINIMUM_COVERAGE, saturation_coverage]``.
    """

    coverage = jnp.clip(
        bubble_number_density * faces.projected_area(volume),
        _MINIMUM_COVERAGE,
        saturation_coverage,
    )
    return -0.25 * (3.0 - coverage) * (1.0 - coverage) - 0.5 * jnp.log(coverage)


@final
class AinscoughGrainGrowth(StrictModule):
    """Ainscough et al. (1973) grain growth to a limiting size.

    BISON Theory Manual (INL/EXT-13-29930 Rev. 3, 2016) Eqs. (8.126)-(8.127):
    ``dD/dt = k (1/D - 1/D_m)`` for the linear-intercept diameter ``D`` with
    ``k = 5.24e7 exp(-2.67e5 / (R T))`` um^2/h and
    ``D_m = 2.23e3 exp(-7620 / T)`` um; the 3D diameter is ``1.56 D``. Growth
    stops at the irradiation-retarded limit (no shrinkage). The Zener pinning
    correction ``g(B) = 1 + 0.002 B`` (B in MWd/kgU) is Eq. (A.7) of Barani's
    2020 thesis appendix; supplied source-bound drag data can replace it.
    """

    rate_prefactor: Array
    activation_energy: Array
    limiting_diameter_prefactor: Array
    limiting_diameter_temperature: Array
    intercept_to_sphere_factor: Array
    burnup_drag_coefficient: Array
    drag_validity: PropertyValidityRange
    drag_source_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        rate_prefactor: ArrayLike = 5.24e7
        * _SQUARE_METER_PER_SECOND_PER_SQUARE_MICROMETER_PER_HOUR,
        activation_energy: ArrayLike = 2.67e5,
        limiting_diameter_prefactor: ArrayLike = 2.23e-3,
        limiting_diameter_temperature: ArrayLike = 7620.0,
        intercept_to_sphere_factor: ArrayLike = 1.56,
        burnup_drag_coefficient: ArrayLike = 0.002,
        drag_validity: PropertyValidityRange | None = None,
        drag_source_id: str = "barani-thesis-2020:eqA.7:zener-pinning",
    ) -> None:
        parameters = host_parameters(
            {
                "rate_prefactor": rate_prefactor,
                "activation_energy": activation_energy,
                "limiting_diameter_prefactor": limiting_diameter_prefactor,
                "limiting_diameter_temperature": limiting_diameter_temperature,
                "intercept_to_sphere_factor": intercept_to_sphere_factor,
                "burnup_drag_coefficient": burnup_drag_coefficient,
            },
            positive=("limiting_diameter_prefactor", "intercept_to_sphere_factor"),
        )
        self.rate_prefactor = jnp.asarray(parameters["rate_prefactor"])
        self.activation_energy = jnp.asarray(parameters["activation_energy"])
        self.limiting_diameter_prefactor = jnp.asarray(
            parameters["limiting_diameter_prefactor"]
        )
        self.limiting_diameter_temperature = jnp.asarray(
            parameters["limiting_diameter_temperature"]
        )
        self.intercept_to_sphere_factor = jnp.asarray(
            parameters["intercept_to_sphere_factor"]
        )
        if not isinstance(drag_source_id, str) or not drag_source_id.strip():
            raise ValueError("Grain-growth drag requires a source_id.")
        if drag_validity is not None and not isinstance(
            drag_validity, PropertyValidityRange
        ):
            raise TypeError("drag_validity must be a PropertyValidityRange.")
        self.burnup_drag_coefficient = jnp.asarray(parameters["burnup_drag_coefficient"])
        self.drag_source_id = drag_source_id
        # Default drag validity: the operating burnup interval, not a fitted range.
        self.drag_validity = (
            PropertyValidityRange(
                {"burnup_mwd_per_kghm": (0.0, 130.0, "MWd/kgHM")},
                source_id=f"{drag_source_id}:declared-operating-interval",
            )
            if drag_validity is None
            else drag_validity
        )
        if self.drag_validity.variables != ("burnup_mwd_per_kghm",):
            raise ValueError(
                "Grain-growth drag validity must declare burnup_mwd_per_kghm."
            )
        self.source_id = (
            "ainscough-1973-jnm-49-117;bison-inl-ext-13-29930-r3:eq8.126-8.127"
        )
        self.law_id = canonical_fingerprint(
            {
                "kind": "ainscough-grain-growth",
                "parameters": parameters,
                "drag_source_id": drag_source_id,
                "drag_validity": self.drag_validity.range_id,
            }
        )

    def limiting_radius(
        self, temperature: ArrayLike, /, *, burnup_mwd_per_kghm: ArrayLike
    ) -> Array:
        """Equivalent-sphere grain radius [m] at the limiting intercept diameter."""

        diameter = self.limiting_diameter_prefactor * jnp.exp(
            -self.limiting_diameter_temperature / jnp.asarray(temperature)
        )
        drag = 1.0 + self.burnup_drag_coefficient * jnp.asarray(burnup_mwd_per_kghm)
        return 0.5 * self.intercept_to_sphere_factor * diameter / drag

    def advance_radius(
        self,
        grain_radius: ArrayLike,
        temperature: ArrayLike,
        time_step: ArrayLike,
        /,
        *,
        burnup_mwd_per_kghm: ArrayLike,
    ) -> Array:
        """Backward-Euler equivalent-sphere grain radius [m] after ``time_step``.

        ``D' - D = k dt (1/D' - 1/D_m)`` is the positive root of
        ``D'^2 - (D - k dt / D_m) D' - k dt = 0``, which lies in ``[D, D_m]``.
        """

        temperature_ = jnp.asarray(temperature)
        radius = jnp.asarray(grain_radius)
        diameter = 2.0 * radius / self.intercept_to_sphere_factor
        limiting = self.limiting_diameter_prefactor * jnp.exp(
            -self.limiting_diameter_temperature / temperature_
        )
        limiting = limiting / (
            1.0 + self.burnup_drag_coefficient * jnp.asarray(burnup_mwd_per_kghm)
        )
        rate_time = (
            self.rate_prefactor
            * jnp.exp(
                -self.activation_energy / (MOLAR_GAS_CONSTANT_J_PER_MOL_K * temperature_)
            )
            * jnp.asarray(time_step)
        )
        root = positive_quadratic_root(diameter - rate_time / limiting, rate_time)
        grown = jnp.where(diameter < limiting, root, diameter)
        # Preserve the physical no-shrink obstacle through the diameter/radius
        # conversion even when the increment is below floating-point spacing.
        return jnp.maximum(radius, 0.5 * self.intercept_to_sphere_factor * grown)


@final
class GrainBoundaryMicrocracking(StrictModule):
    """Barani et al. (2017) transient micro-cracking of grain faces with healing.

    J. Nucl. Mater. 486 (2017) 96, Eqs. (3)-(11): intact fraction
    ``[df/dt]_c = -(dm/dt) f`` with
    ``m(T) = 1 - (1 + Q exp(s (T - T_infl) / B))^(-1/Q)``, ``s = +1`` heating
    and ``-1`` cooling, ``B = 10`` K, ``Q = 33``,
    ``T_infl = 1773 + 520 exp(-bu / 10)`` K (bu in GWd/tU); healing
    ``[df/dt]_h = (du/dt)(1 - f)`` with ``u = bu / 1``. Coverage obeys
    ``dF = F [df]_c`` and saturation coverage ``dF_sat = F_sat ([df]_c + [df]_h)``,
    hence ``F_sat = F_sat,i exp(f - 1)``.
    """

    temperature_span: Array
    asymmetry: Array
    inflection_temperature_limit: Array
    inflection_temperature_amplitude: Array
    inflection_burnup_scale: Array
    healing_burnup_scale: Array
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        temperature_span: ArrayLike = 10.0,
        asymmetry: ArrayLike = 33.0,
        inflection_temperature_limit: ArrayLike = 1773.0,
        inflection_temperature_amplitude: ArrayLike = 520.0,
        inflection_burnup_scale: ArrayLike = 10.0,
        healing_burnup_scale: ArrayLike = 1.0,
    ) -> None:
        parameters = host_parameters(
            {
                "temperature_span": temperature_span,
                "asymmetry": asymmetry,
                "inflection_temperature_limit": inflection_temperature_limit,
                "inflection_temperature_amplitude": inflection_temperature_amplitude,
                "inflection_burnup_scale": inflection_burnup_scale,
                "healing_burnup_scale": healing_burnup_scale,
            },
            positive=(
                "temperature_span",
                "asymmetry",
                "inflection_burnup_scale",
                "healing_burnup_scale",
            ),
        )
        self.temperature_span = jnp.asarray(parameters["temperature_span"])
        self.asymmetry = jnp.asarray(parameters["asymmetry"])
        self.inflection_temperature_limit = jnp.asarray(
            parameters["inflection_temperature_limit"]
        )
        self.inflection_temperature_amplitude = jnp.asarray(
            parameters["inflection_temperature_amplitude"]
        )
        self.inflection_burnup_scale = jnp.asarray(parameters["inflection_burnup_scale"])
        self.healing_burnup_scale = jnp.asarray(parameters["healing_burnup_scale"])
        self.source_id = "barani-2017-jnm-486-96:eq3-11"
        self.law_id = canonical_fingerprint(
            {"kind": "grain-boundary-microcracking", "parameters": parameters}
        )

    def inflection_temperature(self, burnup_mwd_per_kghm: ArrayLike, /) -> Array:
        """``T_infl`` [K]; MWd/kgHM equals the source's GWd/tU for UO2."""

        return (
            self.inflection_temperature_limit
            + self.inflection_temperature_amplitude
            * jnp.exp(-jnp.asarray(burnup_mwd_per_kghm) / self.inflection_burnup_scale)
        )

    def parameter(
        self,
        temperature: ArrayLike,
        inflection_temperature: ArrayLike,
        direction: ArrayLike,
        /,
    ) -> Array:
        """Micro-cracking parameter ``m`` for transient direction ``s = +-1``."""

        exponent = (
            jnp.asarray(direction)
            * (jnp.asarray(temperature) - jnp.asarray(inflection_temperature))
            / self.temperature_span
        )
        # log(1 + Q e^x) evaluated without overflow.
        log_term = jnp.logaddexp(0.0, jnp.log(self.asymmetry) + exponent)
        return 1.0 - jnp.exp(-log_term / self.asymmetry)

    def intactness(
        self,
        intact_fraction: ArrayLike,
        previous_temperature: ArrayLike,
        temperature: ArrayLike,
        burnup_mwd_per_kghm: ArrayLike,
        burnup_increment: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        """Intact fraction after cracking and after cracking plus healing.

        Cracking is exact for the monotone temperature path between the two
        step temperatures (Eq. (5)); ``T_infl`` uses the end-of-step burnup.
        Healing follows Eq. (10) over the burnup increment.
        """

        start = jnp.asarray(previous_temperature)
        end = jnp.asarray(temperature)
        direction = jnp.where(end >= start, 1.0, -1.0)
        inflection = self.inflection_temperature(burnup_mwd_per_kghm)
        increment = self.parameter(end, inflection, direction) - self.parameter(
            start, inflection, direction
        )
        cracked = jnp.asarray(intact_fraction) * jnp.exp(-increment)
        healing = -jnp.expm1(-jnp.asarray(burnup_increment) / self.healing_burnup_scale)
        return cracked, cracked + (1.0 - cracked) * healing


__all__ = [
    "AinscoughGrainGrowth",
    "GrainBoundaryMicrocracking",
    "GrainFaceBubbles",
    "speight_beere_sink_strength",
]
