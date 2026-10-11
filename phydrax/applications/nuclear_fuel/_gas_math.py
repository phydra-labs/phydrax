#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host validation and numerically stable kernels shared by the fuel gas laws."""

from __future__ import annotations

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ...nonlinear._domain_cond import domain_cond


# Acceptance threshold, relative to the magnitude of the checked quantity.
# Reconstructed total-gas and helium concentrations may undershoot zero by
# this roundoff fraction.
NONNEGATIVITY_ROUNDOFF = 64.0 * float(np.finfo(np.float64).eps)


def host_parameters(
    values: Mapping[str, ArrayLike],
    /,
    *,
    positive: tuple[str, ...] = (),
) -> dict[str, float]:
    """Finite real scalars; names in ``positive`` must be positive, others nonnegative."""

    parameters: dict[str, float] = {}
    for name, value in values.items():
        host = np.asarray(value, dtype=np.float64)
        if host.ndim != 0:
            raise ValueError(f"{name} must be a real scalar.")
        number = float(host)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite.")
        if name in positive and not number > 0.0:
            raise ValueError(f"{name} must be positive.")
        if number < 0.0:
            raise ValueError(f"{name} must be nonnegative.")
        parameters[name] = number
    return parameters


def require_site_shape(shape: tuple[int, ...], /) -> None:
    if not isinstance(shape, tuple) or any(
        isinstance(extent, bool) or not isinstance(extent, int) or extent < 0
        for extent in shape
    ):
        raise TypeError("shape must be a tuple of nonnegative ints.")


def positive_grain_radius(grain_radius: ArrayLike, /) -> np.ndarray:
    radius = np.asarray(grain_radius, dtype=np.float64)
    if not np.all(np.isfinite(radius)) or not np.all(radius > 0.0):
        raise ValueError("grain_radius must be finite and positive.")
    return radius


def broadcast_to_sites(value: ArrayLike, shape: tuple[int, ...], /) -> Array:
    return jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float64), shape)


def positive_square_root(value: Array) -> Array:
    """``sqrt`` whose zero branch avoids the unused infinite tangent at zero."""

    def scalar(number: Array) -> Array:
        def positive(_: None) -> Array:
            return jnp.sqrt(number)

        def zero(_: None) -> Array:
            return jnp.zeros_like(number)

        return domain_cond(number > 0.0, positive, zero)

    return jax.vmap(scalar)(value.reshape(-1)).reshape(value.shape)


def positive_quadratic_root(linear: Array, constant: Array, /) -> Array:
    """Positive root of ``x^2 - linear x - constant = 0`` for ``constant >= 0``.

    A negative linear coefficient uses the conjugate form
    ``2 constant / (sqrt(linear^2 + 4 constant) - linear)``, which avoids the
    cancellation of ``linear + sqrt(...)``.
    """

    discriminant = jnp.sqrt(linear**2 + 4.0 * constant)
    denominator = jnp.where(linear < 0.0, discriminant - linear, 1.0)
    return jnp.where(
        linear < 0.0, 2.0 * constant / denominator, 0.5 * (linear + discriminant)
    )


def carnahan_starling_numerator(packing: Array, /) -> Array:
    """``1 + y + y^2 - y^3`` of the hard-sphere compressibility ``Z = (...) / (1 - y)^3``."""

    return 1.0 + packing + packing**2 - packing**3


__all__ = [
    "NONNEGATIVITY_ROUNDOFF",
    "broadcast_to_sites",
    "carnahan_starling_numerator",
    "host_parameters",
    "positive_grain_radius",
    "positive_quadratic_root",
    "positive_square_root",
    "require_site_shape",
]
