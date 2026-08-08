#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array


def normalized_weights(
    count: int,
    /,
    *,
    log_weights: Array | None,
    mask: Array | None,
    rows_valid: Array | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Normalize a masked finite log-weight vector without changing its shape."""
    if log_weights is None:
        values = jnp.zeros((count,), dtype=float)
    else:
        values = jnp.asarray(log_weights, dtype=float)
        if values.shape != (count,):
            raise ValueError(f"log_weights must have shape ({count},).")
    if mask is None:
        included = jnp.ones((count,), dtype=bool)
    else:
        included = jnp.asarray(mask, dtype=bool)
        if included.shape != (count,):
            raise ValueError(f"mask must have shape ({count},).")
    if rows_valid is None:
        finite_rows = jnp.ones((count,), dtype=bool)
    else:
        finite_rows = jnp.asarray(rows_valid, dtype=bool)
        if finite_rows.shape != (count,):
            raise ValueError(f"rows_valid must have shape ({count},).")
    admissible = jnp.isfinite(values) | jnp.isneginf(values)
    valid_inputs = jnp.all(~included | (admissible & finite_rows))
    active = included & finite_rows & jnp.isfinite(values)
    safe_values = jnp.where(active, values, -jnp.inf)
    log_mass = jsp.special.logsumexp(safe_values)
    valid = valid_inputs & jnp.isfinite(log_mass)
    weights = jnp.where(valid & active, jnp.exp(safe_values - log_mass), 0.0)
    return weights, active, valid, log_mass


def log_weights_from_normalized(weights: Array, mask: Array, /) -> Array:
    """Represent normalized nonnegative weights with inactive negative infinities."""
    values = jnp.asarray(weights, dtype=float)
    included = jnp.asarray(mask, dtype=bool)
    safe = jnp.where(included, values, 1.0)
    return jnp.where(included, jnp.log(safe), -jnp.inf)


__all__ = ["log_weights_from_normalized", "normalized_weights"]
