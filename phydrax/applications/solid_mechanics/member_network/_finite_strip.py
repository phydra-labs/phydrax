#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ._cross_section import ThinWalledSection
from ._gbt import compute_gbt_modes, GBTModeBasis


class FiniteStripModeFamily(IntEnum):
    LOCAL = 0
    DISTORTIONAL = 1
    GLOBAL = 2
    INTERACTION = 3


class FiniteStripBucklingProblem(StrictModule):
    section: ThinWalledSection
    young_modulus: Array
    poisson_ratio: Array
    segment_stress: Array
    half_wavelengths: Array
    problem_id: str

    def __init__(
        self,
        section: ThinWalledSection,
        young_modulus: ArrayLike,
        poisson_ratio: ArrayLike,
        segment_stress: ArrayLike,
        half_wavelengths: ArrayLike,
        /,
        *,
        problem_id: str = "finite-strip-buckling",
    ):
        young = jnp.asarray(young_modulus)
        poisson = jnp.asarray(poisson_ratio, dtype=young.dtype)
        stress = jnp.asarray(segment_stress, dtype=young.dtype)
        wavelengths = jnp.asarray(half_wavelengths, dtype=young.dtype)
        if stress.shape != section.thickness.shape or wavelengths.ndim != 1:
            raise ValueError("Finite-strip stress/wavelength arrays have invalid shapes.")
        if bool(
            (young <= 0.0)
            | (poisson <= -1.0)
            | (poisson >= 0.5)
            | jnp.any(wavelengths <= 0.0)
            | jnp.all(stress >= 0.0)
        ):
            raise ValueError("Finite-strip material/loading data are inadmissible.")
        self.section = section
        self.young_modulus = young
        self.poisson_ratio = poisson
        self.segment_stress = stress
        self.half_wavelengths = wavelengths
        self.problem_id = str(problem_id)


class FiniteStripBucklingResult(StrictModule):
    critical_stress: Array
    critical_half_wavelength: Array
    governing_segment: Array
    family: Array
    wavelength_curve: Array
    segment_curves: Array
    gbt_basis: GBTModeBasis
    interaction_margin: Array
    successful: Array


def solve_finite_strip_buckling(
    problem: FiniteStripBucklingProblem,
    /,
) -> FiniteStripBucklingResult:
    """Sweep longitudinal half-wavelengths using plate-strip elastic buckling energy."""
    section = problem.section
    width = section.widths[:, None]
    thickness = section.thickness[:, None]
    wavelength = problem.half_wavelengths[None, :]
    bending = (
        problem.young_modulus * thickness**3 / (12.0 * (1.0 - problem.poisson_ratio**2))
    )
    longitudinal = jnp.pi / wavelength
    transverse = jnp.pi / width
    wave_number = longitudinal**2 + transverse**2
    critical_resultant = bending * wave_number**2 / longitudinal**2
    critical_stress = critical_resultant / thickness
    compression = jnp.maximum(-problem.segment_stress[:, None], 0.0)
    factor = critical_stress / jnp.maximum(
        compression, jnp.finfo(critical_stress.dtype).tiny
    )
    wavelength_curve = jnp.min(factor, axis=0)
    flat_index = jnp.argmin(factor)
    segment_count, wavelength_count = factor.shape
    segment = (flat_index // wavelength_count).astype(jnp.int32)
    wave_index = (flat_index % wavelength_count).astype(jnp.int32)
    basis = compute_gbt_modes(section)
    normalized_wave = problem.half_wavelengths[wave_index] / jnp.max(section.widths)
    family = jnp.where(
        normalized_wave < 2.0,
        int(FiniteStripModeFamily.LOCAL),
        jnp.where(
            normalized_wave < 10.0,
            int(FiniteStripModeFamily.DISTORTIONAL),
            int(FiniteStripModeFamily.GLOBAL),
        ),
    ).astype(jnp.int32)
    sorted_curve = jnp.sort(wavelength_curve)
    interaction = (
        sorted_curve[1] / sorted_curve[0] - 1.0
        if wavelength_curve.size > 1
        else jnp.asarray(jnp.inf, dtype=wavelength_curve.dtype)
    )
    family = jnp.where(
        interaction < 0.05,
        int(FiniteStripModeFamily.INTERACTION),
        family,
    )
    return FiniteStripBucklingResult(
        critical_stress[segment, wave_index],
        problem.half_wavelengths[wave_index],
        segment,
        family,
        wavelength_curve,
        factor,
        basis,
        interaction,
        jnp.isfinite(wavelength_curve[wave_index]) & (wavelength_curve[wave_index] > 0.0),
    )


__all__ = [
    "FiniteStripBucklingProblem",
    "FiniteStripBucklingResult",
    "FiniteStripModeFamily",
    "solve_finite_strip_buckling",
]
