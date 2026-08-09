#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def validate_save_times(
    t0: ArrayLike,
    t1: ArrayLike,
    values: ArrayLike,
    /,
) -> Array:
    """Validate one ordered, finite save schedule inside a time interval."""
    times = jnp.asarray(values, dtype=float)
    if times.ndim != 1 or int(times.shape[0]) <= 0:
        raise ValueError("save_times must be a non-empty rank-1 array.")
    times = eqx.error_if(
        times,
        ~jnp.all(jnp.isfinite(times)),
        "save_times must be finite.",
    )
    if int(times.shape[0]) > 1:
        times = eqx.error_if(
            times,
            ~jnp.all(jnp.diff(times) > 0.0),
            "save_times must be strictly increasing.",
        )
    return eqx.error_if(
        times,
        (times[0] < t0) | (times[-1] > t1),
        "save_times must lie within the problem time interval.",
    )

__all__ = ["validate_save_times"]
