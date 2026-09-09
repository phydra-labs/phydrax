#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Material-sided electrostatics and detailed-balanced interface exchange.

Interface traces are algebraic values, not small-volume bulk cells. All runtime
arguments are SI. An oriented interface runs from its left material to its right
material; particle and energy fluxes are positive in that direction.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...units import derived_unit, KELVIN, METER, SECOND, UnitDefinition
from ._quantities import _positive_scalar, _text, BOLTZMANN_CONSTANT_SI


THERMIONIC_NUMBER_PREFACTOR_UNIT = derived_unit(
    "1/(m2*s*K2)", ((METER, -2), (SECOND, -1), (KELVIN, -2))
)


class InterfaceElectrostaticState(StrictModule):
    """Material-sided potentials, oriented displacement charges and field energy."""

    potential_left: Array
    potential_right: Array
    displacement_left: Array
    displacement_right: Array
    field_energy: Array
    charge_balance_defect: Array
    successful: Array


def interface_electrostatics(
    potential_left: ArrayLike,
    potential_right: ArrayLike,
    permittivity_left: ArrayLike,
    permittivity_right: ArrayLike,
    distance_left: ArrayLike,
    distance_right: ArrayLike,
    area: ArrayLike,
    *,
    sheet_charge: ArrayLike = 0.0,
    potential_jump: ArrayLike = 0.0,
) -> InterfaceElectrostaticState:
    """Eliminate two interface potentials without smearing its sheet charge.

    Distances are from bulk points to the interface, and sheet_charge is the
    integrated physical charge (C), not charge density. The declared dipole jump
    is psi_right_trace - psi_left_trace. This is the exact piecewise-linear
    electrostatic face problem; displacement_right - displacement_left equals
    sheet_charge. No bulk material or band offset is averaged.
    """
    left, right, eps_l, eps_r, dl, dr, area_, charge, jump = jnp.broadcast_arrays(
        *map(
            jnp.asarray,
            (
                potential_left,
                potential_right,
                permittivity_left,
                permittivity_right,
                distance_left,
                distance_right,
                area,
                sheet_charge,
                potential_jump,
            ),
        )
    )
    gl, gr = eps_l * area_ / dl, eps_r * area_ / dr
    trace_l = (gl * left + gr * (right - jump) + charge) / (gl + gr)
    trace_r = trace_l + jump
    displacement_l = gl * (left - trace_l)
    displacement_r = gr * (trace_r - right)
    energy = 0.5 * (gl * (trace_l - left) ** 2 + gr * (right - trace_r) ** 2)
    defect = displacement_r - displacement_l - charge
    successful = (
        (eps_l > 0)
        & (eps_r > 0)
        & (dl > 0)
        & (dr > 0)
        & (area_ > 0)
        & jnp.isfinite(trace_l)
        & jnp.isfinite(trace_r)
        & jnp.isfinite(displacement_l)
        & jnp.isfinite(displacement_r)
        & jnp.isfinite(energy)
        & jnp.isfinite(defect)
    )
    return InterfaceElectrostaticState(
        trace_l, trace_r, displacement_l, displacement_r, energy, defect, successful
    )


class InterfaceExchange(StrictModule):
    """Conservative two-reservoir exchange; heat is not extra lattice heating.

    Number flux is particles/s, energy flux W, and entropy production W/K.
    heat_left and heat_right are heat delivered into their respective reservoirs.
    Their sum is chemical work converted to heat, not an additional copy of the
    transported energy. Charge is obtained with the physical signed carrier charge.
    """

    number_flux: Array
    energy_flux: Array
    heat_left: Array
    heat_right: Array
    entropy_production: Array
    successful: Array

    def number_sources(self) -> tuple[Array, Array]:
        return -self.number_flux, self.number_flux

    def energy_sources(self) -> tuple[Array, Array]:
        return -self.energy_flux, self.energy_flux

    def charge_sources(self, signed_particle_charge: ArrayLike) -> tuple[Array, Array]:
        value = jnp.asarray(signed_particle_charge) * self.number_flux
        return -value, value


def _exponential_difference(log_left, log_right, difference):
    """Use a separately evaluated affinity, including at exact equilibrium."""
    return jnp.where(
        difference >= 0,
        jnp.exp(log_left) * (-jnp.expm1(-jnp.where(difference >= 0, difference, 0))),
        jnp.exp(log_right) * jnp.expm1(jnp.where(difference < 0, difference, 0)),
    )


class ThermionicInterface(StrictModule):
    """Reciprocal 3D Maxwell–Boltzmann thermionic transmission.

    The number-flux Richardson coefficient is explicit, in 1/(m2*s*K2).
    The same transmission spectrum applies in both directions; independently
    fitting two prefactors would violate detailed balance. The transmitting mode
    density is proportional to E-barrier, giving mean crossing energy
    barrier + 2*kB*T. This is a specified nondegenerate interface model, not
    tunneling or a degenerate Fermi interface approximation.

    Particle chemical potential means EFn for electrons and -EFp for holes.
    Barrier energy must use the corresponding particle-energy reference. Forward
    temperature dependence produces real thermoelectric heat/particle exchange;
    equal chemical energies at different temperatures are not equilibrium.
    """

    prefactor: Array
    temperature_range: tuple[float, float] = eqx.field(static=True)
    energy_reference: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    maximum_reduced_chemical_potential: float = eqx.field(static=True)

    def __init__(
        self,
        prefactor,
        *,
        temperature_range: tuple[float, float],
        energy_reference: str,
        provenance: str,
        maximum_reduced_chemical_potential: float = -2.0,
        prefactor_unit: UnitDefinition = THERMIONIC_NUMBER_PREFACTOR_UNIT,
    ):
        self.prefactor = _positive_scalar(
            prefactor,
            prefactor_unit,
            THERMIONIC_NUMBER_PREFACTOR_UNIT,
            "thermionic number prefactor",
        )
        lower = _positive_scalar(
            temperature_range[0], KELVIN, KELVIN, "minimum temperature"
        )
        upper = _positive_scalar(
            temperature_range[1], KELVIN, KELVIN, "maximum temperature"
        )
        if float(upper) < float(lower):
            raise ValueError("Thermionic temperature range must be ordered.")
        if (
            not jnp.isfinite(maximum_reduced_chemical_potential)
            or maximum_reduced_chemical_potential >= 0
        ):
            raise ValueError(
                "A nondegenerate emission model needs an explicit negative eta limit."
            )
        self.temperature_range = (float(lower), float(upper))
        self.energy_reference = _text(energy_reference, "particle energy reference")
        self.provenance = _text(provenance, "thermionic parameter provenance")
        self.maximum_reduced_chemical_potential = float(
            maximum_reduced_chemical_potential
        )

    def evaluate(
        self,
        chemical_potential_left: ArrayLike,
        chemical_potential_right: ArrayLike,
        temperature_left: ArrayLike,
        temperature_right: ArrayLike,
        barrier_energy: ArrayLike,
        area: ArrayLike,
    ) -> InterfaceExchange:
        mu_l, mu_r, tl, tr, barrier, area_ = jnp.broadcast_arrays(
            *map(
                jnp.asarray,
                (
                    chemical_potential_left,
                    chemical_potential_right,
                    temperature_left,
                    temperature_right,
                    barrier_energy,
                    area,
                ),
            )
        )
        k = BOLTZMANN_CONSTANT_SI
        eta_l, eta_r = (mu_l - barrier) / (k * tl), (mu_r - barrier) / (k * tr)
        log_l, log_r = 2 * jnp.log(tl) + eta_l, 2 * jnp.log(tr) + eta_r
        mean_offset = 0.5 * (mu_l + mu_r) - barrier
        affinity = 2 * jnp.log1p((tl - tr) / tr)
        affinity += (mu_l - mu_r) * (tl + tr) / (2 * k * tl * tr)
        affinity -= mean_offset * (tl - tr) / (k * tl * tr)
        particle = (
            area_ * self.prefactor * _exponential_difference(log_l, log_r, affinity)
        )
        forward = area_ * self.prefactor * jnp.exp(log_l)
        reverse = area_ * self.prefactor * jnp.exp(log_r)
        # The emitted half-space spectrum has mean total crossing energy
        # barrier + 2*k*T. This symmetric form preserves exact equilibrium
        # without subtracting two equal total energy streams.
        thermal_energy = k * (tl + tr) * particle
        thermal_energy += k * (tl - tr) * (forward + reverse)
        energy = barrier * particle + thermal_energy
        heat_l = mu_l * particle - energy
        heat_r = energy - mu_r * particle
        entropy = heat_l / tl + heat_r / tr
        entropy_roundoff = (
            64
            * jnp.finfo(entropy.dtype).eps
            * (jnp.abs(heat_l / tl) + jnp.abs(heat_r / tr))
        )
        lower, upper = self.temperature_range
        successful = (
            (tl >= lower)
            & (tl <= upper)
            & (tr >= lower)
            & (tr <= upper)
            & (eta_l <= self.maximum_reduced_chemical_potential)
            & (eta_r <= self.maximum_reduced_chemical_potential)
            & (area_ > 0)
            & jnp.isfinite(particle)
            & jnp.isfinite(energy)
            & jnp.isfinite(heat_l)
            & jnp.isfinite(heat_r)
            & jnp.isfinite(entropy)
            & (entropy >= -entropy_roundoff)
        )
        return InterfaceExchange(particle, energy, heat_l, heat_r, entropy, successful)
