"""Independent host references for the latent Ewald summation (LES).

Plain NumPy/``math`` loops in angstrom, electronvolt and elementary-charge
equivalents, written from the published equations (not from the LES source
or the native implementation):

* :func:`finite_energy` -- direct double loop of the Gaussian pair kernel;
* :func:`mode_energy` -- full-space (not half-space) enumeration of every
  integer reciprocal mode inside ``|k| <= 2 pi / dl`` (optionally inside the
  ``|m_i| <= N_max`` box), with complex structure factors;
* :func:`split_energy` -- the converged Gaussian-charge periodic energy without
  a ``k = 0`` term by an independent Ewald split: an absolutely convergent
  real-space image sum of ``[erf(r/(sqrt2 sigma)) - erf(alpha r)] / r``, a
  reciprocal sum of ``erf(alpha r) / r`` and the explicit ``k = 0`` Fourier
  coefficient ``pi (1/alpha^2 - 2 sigma^2)`` of the short-range part;
* :func:`polarization` and :func:`born_charges` -- the published polarization
  and its finite-difference projected Born charges.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np


NORM_FACTOR = 90.4756
_TWO_PI = 2.0 * math.pi


def _channels(charges: np.ndarray) -> np.ndarray:
    values = np.asarray(charges, dtype=np.float64)
    return values[:, None] if values.ndim == 1 else values


def self_energy(charges: np.ndarray, sigma: float, norm: float = NORM_FACTOR) -> float:
    """``(k_e / 2) phi_G(0) sum q^2`` with ``phi_G(0) = sqrt(2/pi) / sigma``."""
    k_e = norm / _TWO_PI
    return (
        0.5
        * k_e
        * math.sqrt(2.0 / math.pi)
        / sigma
        * float(np.sum(_channels(charges) ** 2))
    )


def finite_energy(
    charges: np.ndarray,
    positions: np.ndarray,
    sigma: float,
    *,
    remove_self: bool = True,
    norm: float = NORM_FACTOR,
) -> float:
    q = _channels(charges)
    r = np.asarray(positions, dtype=np.float64)
    k_e = norm / _TWO_PI
    total = 0.0
    for i in range(r.shape[0]):
        for j in range(i + 1, r.shape[0]):
            distance = float(np.linalg.norm(r[i] - r[j]))
            if distance == 0.0:
                kernel = 2.0 / (math.sqrt(math.pi) * math.sqrt(2.0) * sigma)
            else:
                kernel = math.erf(distance / (math.sqrt(2.0) * sigma)) / distance
            total += k_e * float(np.dot(q[i], q[j])) * kernel
    if not remove_self:
        total += self_energy(q, sigma, norm)
    return total


def mode_energy(
    charges: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    sigma: float,
    dl: float,
    *,
    remove_self: bool = True,
    index_bound: int | None = None,
    norm: float = NORM_FACTOR,
) -> float:
    """``(N / V) sum_{k != 0, k^2 <= (2 pi/dl)^2} exp(-sigma^2 k^2/2)/k^2 |S(k)|^2``."""
    q = _channels(charges)
    r = np.asarray(positions, dtype=np.float64)
    vectors = np.asarray(cell, dtype=np.float64)
    reciprocal = _TWO_PI * np.linalg.inv(vectors).T
    volume = float(np.linalg.det(vectors))
    cutoff = (_TWO_PI / dl) ** 2
    bounds = [int(math.floor(np.linalg.norm(row) / dl)) + 1 for row in vectors]
    if index_bound is not None:
        bounds = [index_bound] * 3
    total = 0.0
    for mode in itertools.product(*(range(-b, b + 1) for b in bounds)):
        if mode == (0, 0, 0):
            continue
        k = np.asarray(mode, dtype=np.float64) @ reciprocal
        squared = float(k @ k)
        if squared > cutoff:
            continue
        structure = np.exp(1j * (r @ k)) @ q
        total += (
            math.exp(-0.5 * sigma**2 * squared)
            / squared
            * float(np.sum(np.abs(structure) ** 2))
        )
    total *= norm / volume
    if remove_self:
        total -= self_energy(q, sigma, norm)
    return total


def split_energy(
    charges: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    sigma: float,
    *,
    alpha: float,
    real_images: int,
    reciprocal_extent: int,
    remove_self: bool = True,
    norm: float = NORM_FACTOR,
) -> float:
    """Converged Gaussian periodic energy without ``k = 0`` (and no background)."""
    q = _channels(charges)
    r = np.asarray(positions, dtype=np.float64)
    vectors = np.asarray(cell, dtype=np.float64)
    volume = float(np.linalg.det(vectors))
    k_e = norm / _TWO_PI
    width = 1.0 / (math.sqrt(2.0) * sigma)
    gram = q @ q.T
    real = 0.0
    for shift in itertools.product(range(-real_images, real_images + 1), repeat=3):
        translation = np.asarray(shift, dtype=np.float64) @ vectors
        for i in range(r.shape[0]):
            for j in range(r.shape[0]):
                distance = float(np.linalg.norm(r[i] - r[j] + translation))
                if distance == 0.0:
                    continue
                real += (
                    gram[i, j]
                    * (math.erf(width * distance) - math.erf(alpha * distance))
                    / distance
                )
    real += float(np.trace(gram)) * 2.0 / math.sqrt(math.pi) * (width - alpha)
    reciprocal_vectors = _TWO_PI * np.linalg.inv(vectors).T
    recip = 0.0
    bound = reciprocal_extent
    for mode in itertools.product(range(-bound, bound + 1), repeat=3):
        if mode == (0, 0, 0):
            continue
        k = np.asarray(mode, dtype=np.float64) @ reciprocal_vectors
        squared = float(k @ k)
        structure = np.exp(1j * (r @ k)) @ q
        recip += (
            4.0
            * math.pi
            * math.exp(-squared / (4.0 * alpha**2))
            / squared
            * float(np.sum(np.abs(structure) ** 2))
        )
    recip /= volume
    charge = np.sum(q, axis=0)
    zero = math.pi * (1.0 / alpha**2 - 2.0 * sigma**2) * float(charge @ charge) / volume
    total = 0.5 * k_e * (recip + real - zero)
    if remove_self:
        total -= self_energy(q, sigma, norm)
    return total


def polarization(
    charges: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray | None,
    *,
    remove_mean: bool,
    epsilon_factor: float,
) -> np.ndarray:
    """Finite ``sum q~ r`` or complex periodic ``sum q~ exp(2 pi i s) / (2 pi i)``."""
    q = np.asarray(charges, dtype=np.float64).reshape(-1)
    if remove_mean:
        q = q - q.mean()
    r = np.asarray(positions, dtype=np.float64)
    norm = math.sqrt(epsilon_factor)
    if cell is None:
        return norm * (q @ r)
    fractional = r @ np.linalg.inv(np.asarray(cell, dtype=np.float64))
    return norm * (q @ np.exp(_TWO_PI * 1j * fractional)) / (_TWO_PI * 1j)


def born_charges(
    charge_function: Callable[[np.ndarray], np.ndarray],
    positions: np.ndarray,
    cell: np.ndarray | None,
    *,
    remove_mean: bool,
    epsilon_factor: float,
    step: float = 1.0e-6,
) -> np.ndarray:
    """Central-difference ``dP_a/dr_ib``, dephased and projected as published."""
    r = np.asarray(positions, dtype=np.float64)
    count = r.shape[0]

    def value(point: np.ndarray) -> np.ndarray:
        return polarization(
            charge_function(point),
            point,
            cell,
            remove_mean=remove_mean,
            epsilon_factor=epsilon_factor,
        )

    jacobian = np.zeros((count, 3, 3), dtype=np.complex128)
    for i in range(count):
        for b in range(3):
            forward = r.copy()
            backward = r.copy()
            forward[i, b] += step
            backward[i, b] -= step
            jacobian[i, :, b] = (value(forward) - value(backward)) / (2.0 * step)
    if cell is None:
        return jacobian.real
    vectors = np.asarray(cell, dtype=np.float64)
    phase = np.exp(_TWO_PI * 1j * (r @ np.linalg.inv(vectors)))
    dephased = (jacobian * np.conj(phase)[:, :, None]).real
    return np.einsum("ac,iab->icb", vectors, dephased)
