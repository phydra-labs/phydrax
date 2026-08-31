#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._types import ImageGeometry2D


RASTER_INACTIVE = 0
RASTER_COMPLETE = 1
RASTER_CLIPPED = 2
RASTER_SUPPORT_OVERFLOW = 3
RASTER_INVALID = 4


class GaussianRasterEvidence(StrictModule):
    """Fixed-capacity support and flux evidence for one rasterization."""

    active: Array
    supported: Array
    truncated: Array
    overflow: Array
    nonfinite: Array
    deposited_flux: Array
    active_count: Array
    supported_count: Array
    truncated_count: Array
    overflow_count: Array
    status: Array


class GaussianRasterResult(StrictModule):
    """A particle image and per-particle rasterization evidence."""

    image: Array
    evidence: GaussianRasterEvidence
    successful: Array
    rasterizer_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)


class GaussianRasterizer(StrictModule, NonTrainableState):
    """Deterministic, bounded-memory Gaussian point-particle rasterizer.

    ``amplitude`` is integrated irradiance over the configured discrete support.
    Image coordinates are ``(row_down, column_right)``. Route indices are
    discrete; derivatives are piecewise derivatives for a fixed route topology.
    """

    maximum_support_radius: int = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    rasterizer_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_support_radius: int = 6,
        *,
        cutoff: float = 3.5,
    ):
        radius = int(maximum_support_radius)
        cutoff_ = float(cutoff)
        if radius < 1:
            raise ValueError("maximum_support_radius must be positive.")
        if not isfinite(cutoff_) or cutoff_ <= 0.0:
            raise ValueError("cutoff must be finite and positive.")
        self.maximum_support_radius = radius
        self.cutoff = cutoff_
        self.rasterizer_id = canonical_fingerprint(
            {
                "kind": "gaussian-particle-rasterizer",
                "maximum_support_radius": radius,
                "cutoff": cutoff_,
                "coordinate_convention": "row-down-column-right",
            }
        )

    def render(
        self,
        geometry: ImageGeometry2D,
        row_column: ArrayLike,
        amplitude: ArrayLike,
        sigma: ArrayLike,
        active: ArrayLike | None = None,
        /,
    ) -> GaussianRasterResult:
        """Render one fixed-capacity particle set into ``geometry``."""
        if not isinstance(geometry, ImageGeometry2D):
            raise TypeError("geometry must be ImageGeometry2D.")
        coordinates = jnp.asarray(row_column)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("row_column must have shape (particle_capacity, 2).")
        if not jnp.issubdtype(coordinates.dtype, jnp.inexact):
            coordinates = coordinates.astype(float)
        capacity = int(coordinates.shape[0])
        amplitudes = jnp.asarray(amplitude, dtype=coordinates.dtype)
        if amplitudes.shape != (capacity,):
            raise ValueError("amplitude must have shape (particle_capacity,).")
        sigmas = jnp.asarray(sigma, dtype=coordinates.dtype)
        if sigmas.ndim == 0:
            sigmas = jnp.broadcast_to(sigmas, (capacity, 2))
        elif sigmas.shape == (capacity,):
            sigmas = jnp.broadcast_to(sigmas[:, None], (capacity, 2))
        elif sigmas.shape == (2,):
            sigmas = jnp.broadcast_to(sigmas, (capacity, 2))
        elif sigmas.shape != (capacity, 2):
            raise ValueError(
                "sigma must be scalar, (2,), (particle_capacity,), or "
                "(particle_capacity, 2)."
            )
        active_ = (
            jnp.ones((capacity,), dtype=bool)
            if active is None
            else jnp.asarray(active, dtype=bool)
        )
        if active_.shape != (capacity,):
            raise ValueError("active must have shape (particle_capacity,).")

        height, width = geometry.image_shape
        dtype = jnp.result_type(coordinates, amplitudes, sigmas)
        image = jnp.zeros((height, width), dtype=dtype)
        radius = self.maximum_support_radius
        offsets = jnp.arange(-radius, radius + 1, dtype=jnp.int32)
        offset_rows, offset_columns = jnp.meshgrid(offsets, offsets, indexing="ij")
        offset_rows = offset_rows.reshape((-1,))
        offset_columns = offset_columns.reshape((-1,))

        def deposit_one(current: Array, particle: tuple[Array, ...]):
            center, flux, spread, enabled = particle
            finite = (
                jnp.all(jnp.isfinite(center))
                & jnp.isfinite(flux)
                & jnp.all(jnp.isfinite(spread))
                & (flux >= 0.0)
                & jnp.all(spread > 0.0)
            )
            usable = enabled & finite
            safe_center = jnp.where(finite, center, jnp.zeros_like(center))
            safe_flux = jnp.where(finite, flux, jnp.zeros_like(flux))
            safe_spread = jnp.where(finite, spread, jnp.ones_like(spread))
            anchor = jnp.floor(safe_center).astype(jnp.int32)
            rows = anchor[0] + offset_rows
            columns = anchor[1] + offset_columns
            delta_row = rows.astype(dtype) - safe_center[0]
            delta_column = columns.astype(dtype) - safe_center[1]
            local_support = (jnp.abs(delta_row) <= self.cutoff * safe_spread[0]) & (
                jnp.abs(delta_column) <= self.cutoff * safe_spread[1]
            )
            inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
            squared_distance = (delta_row / safe_spread[0]) ** 2 + (
                delta_column / safe_spread[1]
            ) ** 2
            weights = jnp.exp(-0.5 * squared_distance) * local_support
            weight_sum = jnp.maximum(jnp.sum(weights), jnp.finfo(dtype).tiny)
            contributions = safe_flux * weights / weight_sum
            routed = usable & local_support & inside
            contributions = jnp.where(routed, contributions, 0.0)
            flat_indices = jnp.clip(rows, 0, height - 1) * width + jnp.clip(
                columns, 0, width - 1
            )
            updated = current.reshape((-1,)).at[flat_indices].add(contributions)
            supported = usable & jnp.any(local_support & inside)
            required_radius = jnp.ceil(self.cutoff * jnp.max(safe_spread)).astype(
                jnp.int32
            )
            overflow = usable & (required_radius > radius)
            border_truncated = usable & jnp.any(local_support & ~inside)
            truncated = overflow | border_truncated
            deposited = jnp.sum(contributions)
            status = jnp.where(
                ~enabled,
                RASTER_INACTIVE,
                jnp.where(
                    ~finite,
                    RASTER_INVALID,
                    jnp.where(
                        overflow,
                        RASTER_SUPPORT_OVERFLOW,
                        jnp.where(truncated, RASTER_CLIPPED, RASTER_COMPLETE),
                    ),
                ),
            )
            evidence = (
                supported,
                truncated,
                overflow,
                enabled & ~finite,
                deposited,
                status,
            )
            return updated.reshape((height, width)), evidence

        image, history = jax.lax.scan(
            deposit_one,
            image,
            (coordinates, amplitudes, sigmas, active_),
        )
        supported, truncated, overflow, nonfinite, deposited_flux, status = history
        evidence = GaussianRasterEvidence(
            active_,
            supported,
            truncated,
            overflow,
            nonfinite,
            deposited_flux,
            jnp.sum(active_, dtype=jnp.int32),
            jnp.sum(supported, dtype=jnp.int32),
            jnp.sum(truncated, dtype=jnp.int32),
            jnp.sum(overflow, dtype=jnp.int32),
            status.astype(jnp.int32),
        )
        return GaussianRasterResult(
            image,
            evidence,
            ~jnp.any(nonfinite | overflow),
            self.rasterizer_id,
            geometry.geometry_id,
        )


def rasterize_gaussians(
    rasterizer: GaussianRasterizer,
    geometry: ImageGeometry2D,
    row_column: ArrayLike,
    amplitude: ArrayLike,
    sigma: ArrayLike,
    active: ArrayLike | None = None,
    /,
) -> GaussianRasterResult:
    """Functional Gaussian rasterization entry point."""
    if not isinstance(rasterizer, GaussianRasterizer):
        raise TypeError("rasterizer must be GaussianRasterizer.")
    return rasterizer.render(geometry, row_column, amplitude, sigma, active)


__all__ = [
    "GaussianRasterEvidence",
    "GaussianRasterResult",
    "GaussianRasterizer",
    "RASTER_CLIPPED",
    "RASTER_COMPLETE",
    "RASTER_INACTIVE",
    "RASTER_INVALID",
    "RASTER_SUPPORT_OVERFLOW",
    "rasterize_gaussians",
]
