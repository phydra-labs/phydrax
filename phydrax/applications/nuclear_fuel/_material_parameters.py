#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Coefficient validation and input coercion of the fuel and cladding laws.

Scalar coefficients are validated on the host as Python floats. Every
material-law input is coerced with ``float64_array``, which leaves float64
arrays unchanged.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike


def finite_parameter(value: ArrayLike, name: str, /) -> float:
    """Return a scalar coefficient as a finite Python float."""
    host = np.asarray(value, dtype=np.float64)
    if host.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    number = float(host)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def positive_parameter(value: ArrayLike, name: str, /) -> float:
    """Return a scalar coefficient as a finite, strictly positive Python float."""
    number = finite_parameter(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def parameter_arrays(parameters: dict[str, float], /) -> dict[str, Array]:
    """Device arrays of validated scalar coefficients, keyed by name."""
    return {name: jnp.asarray(number) for name, number in parameters.items()}


def float64_array(value: ArrayLike, /) -> Array:
    """Material-law input as a float64 array."""
    return jnp.asarray(value, dtype=jnp.float64)


__all__ = [
    "finite_parameter",
    "float64_array",
    "parameter_arrays",
    "positive_parameter",
]
