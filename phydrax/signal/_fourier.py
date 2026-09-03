#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax._spectral._fourier import fourier_resample as _fourier_resample


def fourier_resample(
    values: ArrayLike,
    output_shape: Sequence[int],
    /,
    *,
    axes: Sequence[int] | None = None,
    phase_offsets: Sequence[ArrayLike] | None = None,
) -> Array:
    """Band-limited periodic resampling over explicit or trailing signal axes.

    When ``axes`` is omitted, the trailing ``len(output_shape)`` axes are
    transformed. Unselected axes are independent payload or batch axes.
    """
    array = jnp.asarray(values)
    shape = tuple(int(size) for size in output_shape)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("output_shape must contain positive signal sizes.")
    if axes is None:
        if array.ndim < len(shape):
            raise ValueError(
                "The input rank must be at least the number of output dimensions."
            )
        resolved_axes = tuple(range(array.ndim - len(shape), array.ndim))
    else:
        resolved_axes = tuple(axes)
    return _fourier_resample(
        array,
        shape,
        axes=resolved_axes,
        phase_offsets=phase_offsets,
    )


__all__ = ["fourier_resample"]
