#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-side validation of material-law coefficients and their fingerprint lists."""

from __future__ import annotations

from collections.abc import Sequence
from typing import assert_never, Literal

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike


type CoefficientSign = Literal["any", "nonnegative", "positive"]


def finite_vector(
    values: ArrayLike | Sequence[float],
    name: str,
    size: int,
    /,
    *,
    sign: CoefficientSign = "any",
) -> Array:
    """Float64 device copy of ``size`` finite host values with the required sign."""
    host = np.asarray(values, dtype=np.float64)
    if host.shape != (size,) or not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must be {size} finite values.")
    match sign:
        case "any":
            pass
        case "nonnegative":
            if np.any(host < 0.0):
                raise ValueError(f"{name} must be nonnegative.")
        case "positive":
            if not np.all(host > 0.0):
                raise ValueError(f"{name} must be positive.")
        case _:
            assert_never(sign)
    return jnp.asarray(host)


def finite_scalar(value: ArrayLike, name: str, /, *, positive: bool = False) -> Array:
    """Float64 device copy of one finite host scalar, positive when required."""
    host = np.asarray(value, dtype=np.float64)
    if host.shape != () or not np.isfinite(host) or (positive and not host > 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be one {qualifier} scalar.")
    return jnp.asarray(host)


def host_list(value: Array, /) -> list[float]:
    """Flattened float64 host values, for fingerprint payloads."""
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()
