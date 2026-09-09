# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Qualified 1D density-gradient confinement closure, not open transport."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._strict import StrictModule
from ....units import ONE
from .._quantities import _positive_scalar, _text
from ._basis import _array, HBAR


class DensityGradientResult(StrictModule):
    quantum_potential: Array
    energy_density: Array
    total_energy: Array
    successful: Array


class DensityGradient1D(StrictModule):
    """von Weizsäcker/Bohm closure on one declared orthonormal cell chain.

    For positive volume density ``n``, the correction is
    ``Q = coefficient * H_kin sqrt(n) / sqrt(n)``. ``H_kin`` is the same
    conservative effective-mass operator and homogeneous Dirichlet ghost
    boundary convention as ``EffectiveMass1D``. The corresponding extensive
    gradient energy is the cell-volume integral of ``sqrt(n) H_kin sqrt(n)``.

    The coefficient is explicit because moment-derived density-gradient models
    use closure-dependent factors. This object neither supplies reservoir
    injection nor may be added to an electron population already owned by a
    Schrödinger/NEGF charge model.
    """

    diagonal: Array
    off_diagonal: Array
    cell_volumes: Array
    coefficient: Array
    energy_reference: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        positions,
        effective_mass,
        *,
        area,
        coefficient=1.0,
        energy_reference,
        provenance,
    ):
        x = _array(positions, "density-gradient cell centers")
        if x.ndim != 1 or x.size < 2:
            raise ValueError("DensityGradient1D requires at least two cells.")
        spacing = np.diff(np.asarray(x))
        if np.any(spacing <= 0) or not np.allclose(
            spacing, spacing[0], rtol=1e-9, atol=0
        ):
            raise ValueError("DensityGradient1D admits a strictly uniform 1D chain.")
        mass = _array(
            np.broadcast_to(effective_mass, x.shape),
            "density-gradient effective mass",
            positive=True,
        )
        area_ = _array(area, "density-gradient area", positive=True)
        if area_.shape != ():
            raise ValueError("Density-gradient cross-sectional area must be scalar.")
        dx = spacing[0]
        internal = HBAR**2 / ((mass[:-1] + mass[1:]) * dx**2)
        boundary = jnp.asarray(
            [HBAR**2 / (2 * mass[0] * dx**2), HBAR**2 / (2 * mass[-1] * dx**2)]
        )
        links = jnp.concatenate((boundary[:1], internal, boundary[1:]))
        self.diagonal = links[:-1] + links[1:]
        self.off_diagonal = -internal
        self.cell_volumes = jnp.full(x.shape, dx) * area_
        self.coefficient = _positive_scalar(
            coefficient, ONE, ONE, "density-gradient coefficient"
        )
        self.energy_reference = _text(
            energy_reference, "density-gradient energy reference"
        )
        self.provenance = _text(provenance, "density-gradient provenance")

    def evaluate(self, electron_density):
        density = jnp.asarray(electron_density)
        if density.shape != self.diagonal.shape:
            raise ValueError("Electron density must have one value per closure cell.")
        amplitude = jnp.sqrt(density)
        applied = self.diagonal * amplitude
        applied = applied.at[:-1].add(self.off_diagonal * amplitude[1:])
        applied = applied.at[1:].add(self.off_diagonal * amplitude[:-1])
        potential = self.coefficient * applied / amplitude
        energy_density = self.coefficient * amplitude * applied
        total = jnp.sum(self.cell_volumes * energy_density)
        valid = (
            jnp.all(jnp.isfinite(density))
            & jnp.all(density > 0)
            & jnp.all(jnp.isfinite(potential))
            & jnp.isfinite(total)
            & (total >= -64 * jnp.finfo(total.dtype).eps * jnp.sum(jnp.abs(total)))
        )
        return DensityGradientResult(
            jnp.where(valid, potential, jnp.nan),
            jnp.where(valid, energy_density, jnp.nan),
            jnp.where(valid, total, jnp.nan),
            valid,
        )


__all__ = ["DensityGradient1D", "DensityGradientResult"]
