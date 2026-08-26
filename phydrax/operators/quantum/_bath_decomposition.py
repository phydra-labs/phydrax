#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

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
    if not all(jnp.isfinite(value) for value in (energy, cutoff, thermal)):
        raise ValueError("Drude–Lorentz decomposition parameters must be finite.")
    count = int(term_count)
    if energy < 0.0 or cutoff <= 0.0 or thermal <= 0.0 or count < 1:
        raise ValueError("Drude–Lorentz decomposition parameters are invalid.")
    resonance = cutoff / (2.0 * float(jnp.pi) * thermal)
    nearest = round(resonance)
    if nearest >= 1 and abs(resonance - nearest) <= 1e-12 * max(1.0, resonance):
        raise ValueError("Drude coefficient is singular at a Matsubara resonance.")
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


def _pade_bose_poles_residues(order: int, /) -> tuple[Array, Array]:
    count = int(order)
    alpha = jnp.diag(
        jnp.asarray(
            [
                1.0 / jnp.sqrt((2 * index + 5) * (2 * index + 3))
                for index in range(2 * count - 1)
            ]
        ),
        k=1,
    )
    alpha = alpha + alpha.T
    epsilon = -2.0 / jnp.linalg.eigvalsh(alpha)[:count]
    if count == 1:
        chi = jnp.zeros((0,), dtype=epsilon.dtype)
    else:
        alpha_prime = jnp.diag(
            jnp.asarray(
                [
                    1.0 / jnp.sqrt((2 * index + 7) * (2 * index + 5))
                    for index in range(2 * count - 2)
                ]
            ),
            k=1,
        )
        alpha_prime = alpha_prime + alpha_prime.T
        chi = -2.0 / jnp.linalg.eigvalsh(alpha_prime)[: count - 1]
    prefactor = 0.5 * count * (2 * (count + 1) + 1)
    residues = []
    for index in range(count):
        numerator = prefactor
        for value in chi:
            numerator = numerator * (value**2 - epsilon[index] ** 2)
        denominator = 1.0
        for other in range(count):
            denominator = denominator * (
                epsilon[other] ** 2
                - epsilon[index] ** 2
                + (1.0 if index == other else 0.0)
            )
        residues.append(numerator / denominator)
    return epsilon, jnp.asarray(residues)


def drude_lorentz_pade(
    reorganization_energy: float,
    cutoff_frequency: float,
    temperature: float,
    order: int,
    /,
    *,
    reference_grid_size: int = 256,
) -> BathCorrelationExpansion:
    """Analytic Padé spectrum decomposition in units with hbar = k_B = 1."""
    energy = float(reorganization_energy)
    cutoff = float(cutoff_frequency)
    thermal = float(temperature)
    count = int(order)
    grid_size = int(reference_grid_size)
    if energy < 0.0 or cutoff <= 0.0 or thermal <= 0.0 or count < 1 or grid_size < 2:
        raise ValueError("Drude–Lorentz Padé parameters are invalid.")
    epsilon, kappa = _pade_bose_poles_residues(count)
    frequencies = epsilon * thermal
    if jnp.any(
        jnp.abs(frequencies**2 - cutoff**2) <= 1e-12 * jnp.maximum(1.0, cutoff**2)
    ):
        raise ValueError("A Padé pole collides with the Drude cutoff frequency.")
    coefficients = [energy * cutoff * (1.0 / jnp.tan(cutoff / (2.0 * thermal)) - 1j)]
    coefficients.extend(
        [
            (
                4.0
                * energy
                * cutoff
                * thermal
                * kappa[index]
                * frequencies[index]
                / (frequencies[index] ** 2 - cutoff**2)
            )
            + 0.0j
            for index in range(count)
        ]
    )
    exponents = jnp.concatenate(
        (jnp.asarray([cutoff + 0.0j]), frequencies.astype(complex))
    )
    provisional = BathCorrelationExpansion(
        jnp.asarray(coefficients),
        exponents,
        expansion_id=f"drude-pade:{count}",
    )
    reference = drude_lorentz_matsubara(
        energy,
        cutoff,
        thermal,
        max(256, 16 * count),
    )
    times = jnp.linspace(
        1e-6 / max(cutoff, thermal),
        10.0 / max(cutoff, 1e-12),
        grid_size,
    )
    residual = provisional.residual_against(reference, times)
    return BathCorrelationExpansion(
        provisional.coefficients,
        provisional.exponents,
        expansion_id=provisional.expansion_id,
        fit_residual=residual,
    )


__all__ = [
    "drude_lorentz_matsubara",
    "drude_lorentz_pade",
    "drude_lorentz_pade_from_poles",
    "fit_bath_exponentials",
    "underdamped_brownian_two_pole",
]
