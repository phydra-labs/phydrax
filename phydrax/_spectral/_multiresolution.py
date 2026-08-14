#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


class MultiresolutionCoefficients(StrictModule):
    """Scaling and detail arrays with transform-owned reconstruction metadata."""

    scaling: Array
    details: tuple[tuple[Array, ...], ...]
    reconstruction_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    transform_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        scaling: ArrayLike,
        details: Sequence[Sequence[ArrayLike]],
        /,
        *,
        reconstruction_shapes: Sequence[Sequence[int]],
        transform_fingerprint: str,
    ):
        scaling_array = jnp.asarray(scaling)
        detail_arrays = tuple(
            tuple(jnp.asarray(band) for band in level) for level in details
        )
        shapes = tuple(
            tuple(int(size) for size in shape) for shape in reconstruction_shapes
        )
        fingerprint = str(transform_fingerprint).strip()
        if not detail_arrays:
            raise ValueError("Multiresolution coefficients require at least one level.")
        if len(detail_arrays) != len(shapes):
            raise ValueError(
                "Detail levels and reconstruction shapes must have equal lengths."
            )
        if any(not level for level in detail_arrays):
            raise ValueError("Every multiresolution level requires detail bands.")
        if any(not shape or any(size <= 0 for size in shape) for shape in shapes):
            raise ValueError("Reconstruction shapes must contain positive axis sizes.")
        if not fingerprint:
            raise ValueError("transform_fingerprint must be non-empty.")
        self.scaling = scaling_array
        self.details = detail_arrays
        self.reconstruction_shapes = shapes
        self.transform_fingerprint = fingerprint

    @property
    def levels(self) -> int:
        """Number of multiresolution detail levels."""
        return len(self.details)

    def with_bands(
        self,
        scaling: ArrayLike,
        details: Sequence[Sequence[ArrayLike]],
        /,
    ) -> "MultiresolutionCoefficients":
        """Replace coefficient arrays while preserving reconstruction metadata."""
        return MultiresolutionCoefficients(
            scaling,
            details,
            reconstruction_shapes=self.reconstruction_shapes,
            transform_fingerprint=self.transform_fingerprint,
        )


__all__ = ["MultiresolutionCoefficients"]
