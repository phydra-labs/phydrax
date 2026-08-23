#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

from ._pseudomode import BathCorrelationExpansion


def drude_lorentz_matsubara(
    reorganization_energy: float,
    cutoff_frequency: float,
    temperature: float,
    term_count: int,
    /,
) -> BathCorrelationExpansion:
    """Finite Matsubara decomposition in units with hbar = k_B = 1."""
    energy = float(reorganization_energy)
    cutoff = float(cutoff_frequency)
    thermal = float(temperature)
    count = int(term_count)
    if energy < 0.0 or cutoff <= 0.0 or thermal <= 0.0 or count < 1:
        raise ValueError("Drude–Lorentz decomposition parameters are invalid.")
    coefficients = [energy * cutoff * (1.0 / jnp.tan(cutoff / (2.0 * thermal)) - 1j)]
    exponents = [cutoff + 0.0j]
    for index in range(1, count):
        frequency = 2.0 * jnp.pi * index * thermal
        coefficient = (
            4.0 * energy * cutoff * thermal * frequency / (frequency**2 - cutoff**2)
        )
        coefficients.append(coefficient + 0.0j)
        exponents.append(frequency + 0.0j)
    return BathCorrelationExpansion(
        jnp.asarray(coefficients),
        jnp.asarray(exponents),
        expansion_id=f"drude-matsubara:{count}",
    )


__all__ = ["drude_lorentz_matsubara"]
