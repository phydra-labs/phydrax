#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic normalized Cartesian-to-real-spherical Gaussian transforms."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import sph_harm_y

from ._shell import (
    cartesian_angular_exponents,
    cartesian_primitive_normalization,
    odd_double_factorial,
)


def real_spherical_orders(angular_momentum: int, /) -> tuple[int, ...]:
    angular = int(angular_momentum)
    if angular < 0:
        raise ValueError("Angular momentum must be non-negative.")
    return (0,) + tuple(
        value for order in range(1, angular + 1) for value in (order, -order)
    )


def _same_center_overlap(
    left: tuple[int, int, int], right: tuple[int, int, int], /
) -> float:
    total = tuple(a + b for a, b in zip(left, right, strict=True))
    if any(value % 2 for value in total):
        return 0.0
    exponent_sum = 2.0
    value = 1.0
    for power in total:
        half = power // 2
        value *= (
            odd_double_factorial(2 * half - 1)
            * np.sqrt(np.pi)
            / (2.0**half * exponent_sum ** (half + 0.5))
        )
    left_norm = float(cartesian_primitive_normalization(1.0, left))
    right_norm = float(cartesian_primitive_normalization(1.0, right))
    return left_norm * right_norm * value


def _sphere_points(count: int, /) -> np.ndarray:
    index = np.arange(count, dtype=float)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    azimuth = golden * index
    return np.stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z), axis=1)


def _real_harmonic(
    angular: int,
    order: int,
    polar: np.ndarray,
    azimuth: np.ndarray,
):
    if order == 0:
        return np.real(sph_harm_y(angular, 0, polar, azimuth))
    magnitude = abs(order)
    complex_value = sph_harm_y(angular, magnitude, polar, azimuth)
    phase = (-1.0) ** magnitude * np.sqrt(2.0)
    return phase * (np.real(complex_value) if order > 0 else np.imag(complex_value))


@lru_cache(maxsize=32)
def cartesian_to_real_spherical(angular_momentum: int, /) -> np.ndarray:
    """Return columns of orthonormal real solid harmonics in Cartesian AOs."""

    angular = int(angular_momentum)
    if angular < 0 or angular > 12:
        raise ValueError("Angular momentum must lie in [0, 12].")
    components = cartesian_angular_exponents(angular)
    orders = real_spherical_orders(angular)
    point_count = max(64, 6 * len(components))
    points = _sphere_points(point_count)
    monomials = np.stack(
        tuple(
            points[:, 0] ** x * points[:, 1] ** y * points[:, 2] ** z
            for x, y, z in components
        ),
        axis=1,
    )
    polar = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    spherical = np.stack(
        tuple(_real_harmonic(angular, order, polar, azimuth) for order in orders),
        axis=1,
    )
    raw_coefficients = np.linalg.lstsq(monomials, spherical, rcond=None)[0]
    primitive_norms = np.asarray(
        [float(cartesian_primitive_normalization(1.0, value)) for value in components]
    )
    coefficients = raw_coefficients / primitive_norms[:, None]
    metric = np.asarray(
        [
            [_same_center_overlap(left, right) for right in components]
            for left in components
        ]
    )
    gram = coefficients.T @ metric @ coefficients
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    if np.min(eigenvalues) <= 1.0e-12:
        raise ValueError("Real-spherical transform is numerically rank deficient.")
    inverse_sqrt = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T
    normalized = coefficients @ inverse_sqrt
    residual = np.max(np.abs(normalized.T @ metric @ normalized - np.eye(len(orders))))
    if not np.isfinite(residual) or residual > 1.0e-10:
        raise ValueError("Real-spherical transform failed metric orthonormalization.")
    normalized.setflags(write=False)
    return normalized


__all__ = ["cartesian_to_real_spherical", "real_spherical_orders"]
