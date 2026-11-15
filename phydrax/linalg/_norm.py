#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array


def scaled_norm(
    vector: Array, inner: Callable[[Array, Array], Array] | None = None
) -> Array:
    """Compute a whole-array norm without squaring the unscaled coordinates.

    An explicit Hermitian inner product defines the metric. Scaling protects
    the coordinate arithmetic, but cannot prevent overflow or underflow inside
    an arbitrary inner product (for example, from extreme metric weights).
    """
    # Component magnitudes avoid overflowing complex abs before scaling.
    real = jnp.real(vector)
    scale = jnp.max(jnp.abs(real), initial=0.0)
    if jnp.iscomplexobj(vector):
        imaginary = jnp.imag(vector)
        scale = jnp.maximum(scale, jnp.max(jnp.abs(imaginary), initial=0.0))
    scale = jax.lax.stop_gradient(scale)
    # A nonfinite coordinate must reach the inner product rather than being
    # hidden by division by infinity; zero similarly needs a safe divisor.
    divisor = jnp.where((scale == 0) | ~jnp.isfinite(scale), 1.0, scale)
    if jnp.iscomplexobj(vector):
        normalized = jax.lax.complex(real / divisor, imaginary / divisor)
    else:
        normalized = vector / divisor
    squared = jnp.real(
        jnp.vdot(normalized, normalized)
        if inner is None
        else inner(normalized, normalized)
    )
    squared = jnp.maximum(squared, 0.0)
    zero = squared == 0.0
    # Avoid sqrt's singular derivative at exact zero in both JVP and VJP.
    magnitude = jnp.where(zero, 0.0, jnp.sqrt(jnp.where(zero, 1.0, squared)))
    return divisor * magnitude
