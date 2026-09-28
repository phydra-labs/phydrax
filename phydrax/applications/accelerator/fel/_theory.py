#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""High-gain FEL scaling: Pierce parameter, 1-D and Ming Xie gain lengths.

With coupling ``κ = a_w [JJ]₁ / √2`` (planar ``a_w = K/√2``; helical
``a_w = K``, ``[JJ]₁ = 1``) and peak density ``n_e = I / (e c 2π σ_x σ_y)``,
the Pierce parameter of the period-averaged equations is

``ρ³ = e² κ² n_e / (8 ε₀ mₑ c² γ³ k_u²) = (a_w [JJ] ω_p / (4 c k_u))² / γ³``,

the cold 1-D field amplitude grows as ``exp(√3 ρ k_u z)`` and the 1-D power
gain length is ``L_1D = λ_u / (4π√3 ρ)``. Ming Xie's fit (Xie, PAC 1995, p.
183; NIM A 445, 59, 2000) gives ``L_g = L_1D (1 + Λ(η_d, η_ε, η_γ))`` with
``η_d = L_1D / (2kσ²)``, ``η_ε = (L_1D/β)(4πε/λ)``, ``η_γ = 4π (L_1D/λ_u) σ_δ``
and the saturation estimate ``P_sat ≈ 1.6 ρ (L_1D/L_g)² P_beam``. The fit
reproduces the exact 3-D eigenmode growth rate within about ten percent over
its fitted domain; it assumes a round, matched beam in smooth focusing.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array

from ...._strict import StrictModule
from ...._validation import positive_finite_float
from ....typing import Float64
from ._lattice import FELUndulatorLattice
from ._slices import FELBeamSlices
from ._types import FELSliceDim


# (coefficient, η_d exponent, η_ε exponent, η_γ exponent) of Xie's Λ fit.
_MING_XIE_TERMS = (
    (0.45, 0.57, 0.0, 0.0),
    (0.55, 0.0, 1.6, 0.0),
    (3.0, 0.0, 0.0, 2.0),
    (0.35, 0.0, 2.9, 2.4),
    (51.0, 0.95, 0.0, 3.0),
    (5.4, 0.7, 1.9, 0.0),
    (1140.0, 2.2, 2.9, 3.2),
)
_MING_XIE_SATURATION_COEFFICIENT = 1.6


class FELScalingEstimate(StrictModule):
    """Per-slice analytic high-gain scaling (scale units)."""

    __strict_contract__ = True

    pierce_parameter: Float64[FELSliceDim]
    one_dimensional_gain_length: Float64[FELSliceDim]
    diffraction_parameter: Float64[FELSliceDim]
    emittance_parameter: Float64[FELSliceDim]
    energy_spread_parameter: Float64[FELSliceDim]
    ming_xie_gain_length: Float64[FELSliceDim]
    ming_xie_saturation_power: Float64[FELSliceDim]
    beam_power: Float64[FELSliceDim]
    beam_area: Float64[FELSliceDim]


def slice_beam_area(slices: FELBeamSlices, /) -> Array:
    """``2π σ_x σ_y`` with ``σ² = ε_n β / γ`` per plane."""
    geometric = slices.normalized_emittances / slices.lorentz_factors[:, None]
    sizes = jnp.sqrt(geometric * slices.beta_functions)
    return 2.0 * jnp.pi * sizes[:, 0] * sizes[:, 1]


def fel_scaling_estimate(
    lattice: FELUndulatorLattice,
    slices: FELBeamSlices,
    wavelength: float,
    /,
    *,
    segment: int = 0,
) -> FELScalingEstimate:
    """Evaluate Pierce, 1-D, and Ming Xie scaling for every slice.

    ``segment`` selects the undulator module whose period and deflection enter
    the estimate; ``wavelength`` is the radiation wavelength (scale length).
    """
    if not isinstance(lattice, FELUndulatorLattice):
        raise TypeError("lattice must be an FELUndulatorLattice.")
    if not isinstance(slices, FELBeamSlices):
        raise TypeError("slices must be FELBeamSlices.")
    radiation = positive_finite_float(wavelength, "wavelength")
    scale = lattice.scale
    charge = float(scale.elementary_charge)
    mass = float(scale.electron_mass)
    light = float(scale.speed_of_light)
    permittivity = float(scale.vacuum_permittivity)
    device = lattice.segments[segment].device
    coupling = lattice.fundamental_couplings[segment]
    undulator_wavenumber = 2.0 * math.pi / device.period
    wavenumber = 2.0 * math.pi / radiation

    gamma = slices.lorentz_factors
    area = slice_beam_area(slices)
    density = slices.currents / (charge * light * area)
    pierce = jnp.cbrt(
        charge**2
        * coupling**2
        * density
        / (8.0 * permittivity * mass * light**2 * gamma**3 * undulator_wavenumber**2)
    )
    gain_1d = 1.0 / (2.0 * math.sqrt(3.0) * pierce * undulator_wavenumber)
    geometric = slices.normalized_emittances / gamma[:, None]
    emittance = jnp.sqrt(geometric[:, 0] * geometric[:, 1])
    size_squared = area / (2.0 * jnp.pi)
    beta = size_squared / emittance
    diffraction = gain_1d / (2.0 * wavenumber * size_squared)
    emittance_parameter = gain_1d / beta * 4.0 * jnp.pi * emittance / radiation
    spread = 4.0 * jnp.pi * gain_1d / device.period * slices.relative_energy_spreads
    total = jnp.zeros_like(gain_1d)
    for coefficient, diffraction_power, emittance_power, spread_power in _MING_XIE_TERMS:
        total = total + coefficient * (
            jnp.power(diffraction, diffraction_power)
            * jnp.power(emittance_parameter, emittance_power)
            * jnp.power(spread, spread_power)
        )
    gain_ming_xie = gain_1d * (1.0 + total)
    beam_power = gamma * mass * light**2 * slices.currents / charge
    saturation = (
        _MING_XIE_SATURATION_COEFFICIENT
        * pierce
        * (gain_1d / gain_ming_xie) ** 2
        * beam_power
    )
    return FELScalingEstimate(
        pierce_parameter=pierce,
        one_dimensional_gain_length=gain_1d,
        diffraction_parameter=diffraction,
        emittance_parameter=emittance_parameter,
        energy_spread_parameter=spread,
        ming_xie_gain_length=gain_ming_xie,
        ming_xie_saturation_power=saturation,
        beam_power=beam_power,
        beam_area=area,
    )


__all__ = ["FELScalingEstimate", "fel_scaling_estimate", "slice_beam_area"]
