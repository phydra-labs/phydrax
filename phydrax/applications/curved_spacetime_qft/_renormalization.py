#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import phydrax.ein as ein

from ..._strict import StrictModule
from ._modes import (
    adiabatic_frequencies,
    differentiate_time,
    ModeEvolutionEvidence,
    PreparedFLRWModes,
)


class RenormalizedStressEvidence(StrictModule):
    raw_energy_density: Array
    raw_pressure: Array
    subtraction_energy_density: Array
    subtraction_pressure: Array
    renormalized_energy_density: Array
    renormalized_pressure: Array
    continuity_residual: Array
    relative_continuity_residual: Array
    maximum_relative_continuity_residual: Array
    finite: Array
    conserved: Array
    adiabatic_order: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def _mode_stress_integrands(
    prepared: PreparedFLRWModes,
    modes: Array,
    derivatives: Array,
    /,
) -> tuple[Array, Array]:
    scale = prepared.plan.scale_factors[:, None]
    conformal_hubble = (prepared.plan.scale_factor_primes / prepared.plan.scale_factors)[
        :, None
    ]
    covariant_derivative = derivatives - conformal_hubble * modes
    kinetic = jnp.abs(covariant_derivative) ** 2
    amplitudes = jnp.abs(modes) ** 2
    momentum_squared = prepared.plan.comoving_wavenumbers[None, :] ** 2
    mass_squared = (scale * prepared.plan.mass) ** 2
    energy = 0.5 * (kinetic + (momentum_squared + mass_squared) * amplitudes) / scale**4
    pressure = (
        0.5 * (kinetic - (momentum_squared / 3.0 + mass_squared) * amplitudes) / scale**4
    )
    return energy, pressure


def adiabatic_hadamard_subtraction(
    prepared: PreparedFLRWModes,
    evolution: ModeEvolutionEvidence,
    /,
    *,
    order: int = 4,
    conservation_tolerance: float = 1e-6,
) -> RenormalizedStressEvidence:
    """Subtract the finite-cutoff homogeneous adiabatic/Hadamard parametrix."""
    if not isinstance(prepared, PreparedFLRWModes):
        raise TypeError("prepared must be PreparedFLRWModes.")
    if not isinstance(evolution, ModeEvolutionEvidence):
        raise TypeError("evolution must be ModeEvolutionEvidence.")
    if prepared.plan.curvature_coupling != 0.0:
        raise ValueError(
            "Canonical stress subtraction currently requires minimal curvature coupling."
        )
    order_ = int(order)
    tolerance = float(conservation_tolerance)
    if order_ not in (0, 2, 4):
        raise ValueError("adiabatic order must be zero, two, or four.")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("conservation_tolerance must be finite and nonnegative.")
    expected = prepared.frequency_squared.shape
    if evolution.modes.shape != expected or evolution.derivatives.shape != expected:
        raise ValueError("Evolution history does not match the prepared background.")
    frequencies = adiabatic_frequencies(prepared, order_)
    frequency_primes = differentiate_time(frequencies, prepared.plan.conformal_times)
    subtraction_modes = 1.0 / jnp.sqrt(2.0 * frequencies)
    subtraction_derivatives = (
        -0.5 * frequency_primes / frequencies - 1.0j * frequencies
    ) * subtraction_modes
    raw_energy, raw_pressure = _mode_stress_integrands(
        prepared, evolution.modes, evolution.derivatives
    )
    subtraction_energy, subtraction_pressure = _mode_stress_integrands(
        prepared,
        subtraction_modes.astype(evolution.modes.dtype),
        subtraction_derivatives.astype(evolution.derivatives.dtype),
    )
    measure = (
        prepared.plan.momentum_weights
        * prepared.plan.comoving_wavenumbers**2
        / (2.0 * jnp.pi**2)
    )
    raw_energy_density = ein.contract("k,tk->t", measure, raw_energy)
    raw_pressure_density = ein.contract("k,tk->t", measure, raw_pressure)
    subtraction_energy_density = ein.contract("k,tk->t", measure, subtraction_energy)
    subtraction_pressure_density = ein.contract("k,tk->t", measure, subtraction_pressure)
    renormalized_energy = raw_energy_density - subtraction_energy_density
    renormalized_pressure = raw_pressure_density - subtraction_pressure_density
    energy_prime = differentiate_time(renormalized_energy, prepared.plan.conformal_times)
    conformal_hubble = prepared.plan.scale_factor_primes / prepared.plan.scale_factors
    expansion = 3.0 * conformal_hubble * (renormalized_energy + renormalized_pressure)
    continuity = energy_prime + expansion
    continuity_scale = jnp.maximum(1.0, jnp.abs(energy_prime) + jnp.abs(expansion))
    relative = jnp.abs(continuity) / continuity_scale
    maximum = jnp.max(relative)
    finite = (
        jnp.all(jnp.isfinite(renormalized_energy))
        & jnp.all(jnp.isfinite(renormalized_pressure))
        & jnp.all(jnp.isfinite(relative))
    )
    return RenormalizedStressEvidence(
        raw_energy_density=raw_energy_density,
        raw_pressure=raw_pressure_density,
        subtraction_energy_density=subtraction_energy_density,
        subtraction_pressure=subtraction_pressure_density,
        renormalized_energy_density=renormalized_energy,
        renormalized_pressure=renormalized_pressure,
        continuity_residual=continuity,
        relative_continuity_residual=relative,
        maximum_relative_continuity_residual=maximum,
        finite=finite,
        conserved=finite & (maximum <= tolerance),
        adiabatic_order=order_,
        prepared_id=prepared.prepared_id,
        claim=(
            "finite-cutoff-minimally-coupled-flrw-adiabatic-hadamard-reference-only;"
            "no-covariant-renormalization-or-continuum-limit-claim"
        ),
    )


__all__ = ["RenormalizedStressEvidence", "adiabatic_hadamard_subtraction"]
