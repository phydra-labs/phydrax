"""Small independent host references for long-range atomistic interactions.

Pure NumPy/``math`` implementations that share no code with the native terms:
explicitly shell-converged Ewald sums under a stated macroscopic convention,
two-dimensionally periodic (slab) Ewald, finite-cutoff DSF and Yukawa sums,
direct absolutely convergent dispersion lattice sums, the exact periodic
structure factor and central finite differences. Every routine is a bounded
reference for a handful of atoms; none is a production method.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
from scipy import special


#: Rock-salt Madelung constant referred to the nearest-neighbor distance.
NACL_MADELUNG = 1.747564594633182
#: Caesium-chloride Madelung constant referred to the nearest-neighbor distance.
CSCL_MADELUNG = 1.762674773070


def _erfc(value: np.ndarray) -> np.ndarray:
    return special.erfc(np.asarray(value, dtype=np.float64))


def _shifts(extents: tuple[int, ...]) -> np.ndarray:
    return np.asarray(
        list(itertools.product(*(range(-extent, extent + 1) for extent in extents))),
        dtype=np.float64,
    )


def ewald_energy(
    positions: np.ndarray,
    charges: np.ndarray,
    vectors: np.ndarray,
    alpha: float,
    /,
    *,
    real_shells: int,
    reciprocal_shells: int,
    background: bool = False,
    coulomb_constant: float = 1.0,
) -> float:
    """Tin-foil Ewald energy with explicit real-space image shells.

    Real space sums ``erfc(alpha r)/r`` over every image ``|n_i| <= real_shells``
    including nonzero self images; reciprocal space sums every mode
    ``0 < |m_i| <= reciprocal_shells``. ``background`` adds the uniform
    neutralizing-background term ``-pi Q^2 / (2 alpha^2 V)``.
    """
    r = np.asarray(positions, dtype=np.float64)
    q = np.asarray(charges, dtype=np.float64)
    h = np.asarray(vectors, dtype=np.float64)
    volume = abs(float(np.linalg.det(h)))
    translations = _shifts((real_shells,) * 3) @ h
    real = 0.0
    for i in range(r.shape[0]):
        for j in range(r.shape[0]):
            d = r[i] - r[j] + translations
            distance = np.sqrt(np.sum(d * d, axis=-1))
            keep = distance > 1.0e-12
            real += (
                0.5
                * q[i]
                * q[j]
                * float(np.sum(_erfc(alpha * distance[keep]) / distance[keep]))
            )
    reciprocal_rows = 2.0 * np.pi * np.linalg.inv(h).T
    modes = _shifts((reciprocal_shells,) * 3)
    modes = modes[np.any(modes != 0.0, axis=-1)]
    k = modes @ reciprocal_rows
    k2 = np.sum(k * k, axis=-1)
    phase = k @ r.T
    structure = (q[None, :] * np.exp(1j * phase)).sum(axis=-1)
    reciprocal = (
        2.0
        * np.pi
        / volume
        * float(np.sum(np.exp(-k2 / (4.0 * alpha**2)) / k2 * np.abs(structure) ** 2))
    )
    self_energy = -alpha / math.sqrt(math.pi) * float(np.sum(q * q))
    total = real + reciprocal + self_energy
    if background:
        total += -math.pi * float(np.sum(q)) ** 2 / (2.0 * alpha**2 * volume)
    return coulomb_constant * total


def converged_ewald_energy(
    positions: np.ndarray,
    charges: np.ndarray,
    vectors: np.ndarray,
    alpha: float,
    /,
    *,
    tolerance: float = 1.0e-10,
    background: bool = False,
    maximum_shells: int = 12,
    real_shell_limit: int = 3,
) -> tuple[float, list[float]]:
    """Increase the shell counts until two successive energies agree.

    The real-space shell count follows the reciprocal one up to
    ``real_shell_limit`` (``erfc`` decays far faster than a cell). Returns the
    converged energy and the full shell history as evidence; raises if
    ``maximum_shells`` is reached first.
    """
    history: list[float] = []
    for shells in range(2, maximum_shells + 1):
        history.append(
            ewald_energy(
                positions,
                charges,
                vectors,
                alpha,
                real_shells=min(shells, real_shell_limit),
                reciprocal_shells=shells,
                background=background,
            )
        )
        if len(history) >= 2 and abs(history[-1] - history[-2]) <= tolerance:
            return history[-1], history
    raise RuntimeError(f"Ewald reference did not converge: {history}")


def slab_ewald2d_energy(
    positions: np.ndarray,
    charges: np.ndarray,
    in_plane: np.ndarray,
    alpha: float,
    /,
    *,
    real_shells: int,
    reciprocal_shells: int,
) -> float:
    """Neutral two-dimensionally periodic Ewald energy (Parry/Heyes form).

    ``in_plane`` holds two row vectors in the ``xy`` plane; ``z`` is the
    nonperiodic coordinate. Valid for neutral sets only; the result is
    independent of ``alpha`` when converged, which callers should verify.
    """
    r = np.asarray(positions, dtype=np.float64)
    q = np.asarray(charges, dtype=np.float64)
    a = np.asarray(in_plane, dtype=np.float64)
    if abs(float(np.sum(q))) > 1.0e-12:
        raise ValueError("The 2D Ewald reference is defined for neutral sets.")
    area = abs(a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0])
    lattice = np.zeros((2, 3))
    lattice[:, :2] = a[:, :2]
    translations = _shifts((real_shells, real_shells)) @ lattice
    planar = np.asarray([[a[0, 0], a[0, 1]], [a[1, 0], a[1, 1]]])
    reciprocal_rows = 2.0 * np.pi * np.linalg.inv(planar).T
    modes = _shifts((reciprocal_shells, reciprocal_shells))
    modes = modes[np.any(modes != 0.0, axis=-1)]
    g = modes @ reciprocal_rows
    gn = np.sqrt(np.sum(g * g, axis=-1))
    total = 0.0
    for i in range(r.shape[0]):
        for j in range(r.shape[0]):
            pair = q[i] * q[j]
            d = r[i] - r[j] + translations
            distance = np.sqrt(np.sum(d * d, axis=-1))
            keep = distance > 1.0e-12
            real = float(np.sum(_erfc(alpha * distance[keep]) / distance[keep]))
            rho = r[i, :2] - r[j, :2]
            z = r[i, 2] - r[j, 2]
            first = np.exp(gn * z) * _erfc(gn / (2.0 * alpha) + alpha * z)
            second = np.exp(-gn * z) * _erfc(gn / (2.0 * alpha) - alpha * z)
            reciprocal = (
                math.pi / area * float(np.sum(np.cos(g @ rho) / gn * (first + second)))
            )
            zero = (
                -2.0
                * math.pi
                / area
                * (
                    z * math.erf(alpha * z)
                    + math.exp(-((alpha * z) ** 2)) / (alpha * math.sqrt(math.pi))
                )
            )
            total += 0.5 * pair * (real + reciprocal + zero)
    return total - alpha / math.sqrt(math.pi) * float(np.sum(q * q))


def _minimum_image(displacement: np.ndarray, vectors: np.ndarray | None) -> np.ndarray:
    if vectors is None:
        return displacement
    h = np.asarray(vectors, dtype=np.float64)
    candidates = displacement + _shifts((2, 2, 2)) @ h
    return candidates[np.argmin(np.sum(candidates * candidates, axis=-1))]


def pair_sum(
    positions: np.ndarray,
    charges: np.ndarray,
    kernel: Callable[[float], float],
    cutoff: float,
    /,
    *,
    vectors: np.ndarray | None = None,
) -> float:
    """``sum_{i<j, r<cutoff} q_i q_j kernel(r)`` with minimum-image pairs."""
    r = np.asarray(positions, dtype=np.float64)
    q = np.asarray(charges, dtype=np.float64)
    total = 0.0
    for i in range(r.shape[0]):
        for j in range(i + 1, r.shape[0]):
            d = _minimum_image(r[i] - r[j], vectors)
            distance = float(np.sqrt(np.sum(d * d)))
            if distance < cutoff:
                total += q[i] * q[j] * kernel(distance)
    return total


def dsf_energy(
    positions: np.ndarray,
    charges: np.ndarray,
    alpha: float,
    cutoff: float,
    /,
    *,
    vectors: np.ndarray | None = None,
) -> float:
    """Fennell-Gezelter damped shifted-force energy including its self term."""
    shift = math.erfc(alpha * cutoff) / cutoff
    slope = (
        shift / cutoff
        + 2.0 * alpha / math.sqrt(math.pi) * math.exp(-((alpha * cutoff) ** 2)) / cutoff
    )

    def kernel(distance: float) -> float:
        return (
            math.erfc(alpha * distance) / distance - shift + slope * (distance - cutoff)
        )

    q = np.asarray(charges, dtype=np.float64)
    self_energy = -(0.5 * shift + alpha / math.sqrt(math.pi)) * float(np.sum(q * q))
    return pair_sum(positions, charges, kernel, cutoff, vectors=vectors) + self_energy


def yukawa_energy(
    positions: np.ndarray,
    charges: np.ndarray,
    kappa: float,
    cutoff: float,
    truncation: str,
    /,
    *,
    vectors: np.ndarray | None = None,
) -> float:
    """Screened Coulomb ``exp(-kappa r)/r`` under a named truncation."""

    def phi(distance: float) -> float:
        return math.exp(-kappa * distance) / distance

    def derivative(distance: float) -> float:
        return -math.exp(-kappa * distance) * (kappa * distance + 1.0) / distance**2

    def kernel(distance: float) -> float:
        value = phi(distance)
        if truncation == "hard":
            return value
        value -= phi(cutoff)
        if truncation == "shifted-potential":
            return value
        return value - derivative(cutoff) * (distance - cutoff)

    return pair_sum(positions, charges, kernel, cutoff, vectors=vectors)


def dispersion_lattice_energy(
    positions: np.ndarray,
    c6: np.ndarray,
    vectors: np.ndarray,
    /,
    *,
    shells: int,
) -> float:
    """``-1/2 sum'_{i,j,n} C6_ij / |r_i - r_j + n H|^6`` (absolutely convergent)."""
    r = np.asarray(positions, dtype=np.float64)
    c = np.asarray(c6, dtype=np.float64)
    translations = _shifts((shells,) * 3) @ np.asarray(vectors, dtype=np.float64)
    total = 0.0
    for i in range(r.shape[0]):
        for j in range(r.shape[0]):
            d = r[i] - r[j] + translations
            squared = np.sum(d * d, axis=-1)
            keep = squared > 1.0e-24
            total -= 0.5 * c[i, j] * float(np.sum(squared[keep] ** -3))
    return total


def structure_factor(
    positions: np.ndarray, values: np.ndarray, wavevector: np.ndarray, /
) -> complex:
    """Exact ``sum_i v_i exp(i k . r_i)``."""
    phase = np.asarray(positions, dtype=np.float64) @ np.asarray(wavevector)
    return complex(np.sum(np.asarray(values) * np.exp(1j * phase)))


def central_difference(
    function: Callable[[np.ndarray], float], point: np.ndarray, step: float, /
) -> np.ndarray:
    """Central finite-difference gradient of a scalar host function."""
    base = np.asarray(point, dtype=np.float64)
    gradient = np.zeros_like(base)
    for index in np.ndindex(base.shape):
        forward = base.copy()
        backward = base.copy()
        forward[index] += step
        backward[index] -= step
        gradient[index] = (function(forward) - function(backward)) / (2.0 * step)
    return gradient
