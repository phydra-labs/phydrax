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
    for index in range(1, count):
        frequency = 2.0 * jnp.pi * index * thermal
        if abs(float(frequency**2 - cutoff**2)) <= 1e-12 * max(1.0, cutoff**2):
            raise ValueError("Matsubara pole collides with the Drude cutoff frequency.")
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


def fit_bath_exponentials(
    target,
    times,
    exponents,
    /,
    *,
    expansion_id: str,
) -> BathCorrelationExpansion:
    """Fit coefficients for caller-supplied stable rational/Padé poles."""
    if not callable(target):
        raise TypeError("target must be callable.")
    times_ = jnp.asarray(times, dtype=float)
    exponents_ = jnp.asarray(exponents, dtype=complex)
    if times_.ndim != 1 or exponents_.ndim != 1:
        raise ValueError("Bath fit times and exponents must be vectors.")
    if jnp.any(jnp.real(exponents_) <= 0.0):
        raise ValueError("Bath fit exponents must have positive real part.")
    design = jnp.exp(-times_[:, None] * exponents_[None, :])
    reference = jnp.asarray(target(times_), dtype=complex)
    coefficients = jnp.linalg.lstsq(design, reference, rcond=None)[0]
    residual = jnp.sqrt(jnp.mean(jnp.abs(design @ coefficients - reference) ** 2))
    return BathCorrelationExpansion(
        coefficients,
        exponents_,
        expansion_id=expansion_id,
        fit_residual=residual,
    )


def underdamped_brownian_two_pole(
    frequency: float,
    damping: float,
    coupling: float,
    /,
) -> BathCorrelationExpansion:
    """Two-pole underdamped Brownian correlation approximation."""
    omega = float(frequency)
    gamma = float(damping)
    strength = float(coupling)
    if omega <= 0.0 or gamma <= 0.0 or gamma >= 2.0 * omega:
        raise ValueError("Underdamped Brownian parameters require 0 < gamma < 2 omega.")
    damped = jnp.sqrt(omega**2 - 0.25 * gamma**2)
    exponents = jnp.asarray([0.5 * gamma + 1j * damped, 0.5 * gamma - 1j * damped])
    coefficients = jnp.asarray([0.5 * strength, 0.5 * strength], dtype=complex)
    return BathCorrelationExpansion(
        coefficients,
        exponents,
        expansion_id="underdamped-brownian-two-pole",
    )


def drude_lorentz_pade_from_poles(
    reorganization_energy: float,
    cutoff_frequency: float,
    temperature: float,
    pade_poles,
    reference_times,
    /,
    *,
    reference_matsubara_terms: int = 128,
) -> BathCorrelationExpansion:
    """Fit a Drude correlation using caller-supplied positive Padé poles."""
    reference = drude_lorentz_matsubara(
        reorganization_energy,
        cutoff_frequency,
        temperature,
        reference_matsubara_terms,
    )
    poles = jnp.asarray(pade_poles, dtype=complex)
    if poles.ndim != 1 or jnp.any(jnp.real(poles) <= 0.0):
        raise ValueError("Padé poles must be a vector with positive real parts.")
    exponents = jnp.concatenate((jnp.asarray([cutoff_frequency + 0.0j]), poles))
    return fit_bath_exponentials(
        reference,
        reference_times,
        exponents,
        expansion_id=f"drude-pade-supplied-poles:{poles.shape[0]}",
    )


__all__ = [
    "drude_lorentz_matsubara",
    "drude_lorentz_pade_from_poles",
    "fit_bath_exponentials",
    "underdamped_brownian_two_pole",
]
