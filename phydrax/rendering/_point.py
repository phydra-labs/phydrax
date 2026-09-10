#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..imaging import ImagePlaneSupport
from ..sparse import EdgeRelation, RelationExecutionPlan


RASTER_INACTIVE = 0
RASTER_COMPLETE = 1
RASTER_CLIPPED = 2
RASTER_SUPPORT_OVERFLOW = 3
RASTER_INVALID = 4

GaussianRasterKind = Literal["reference", "tiled"]
GaussianRasterAccumulation = Literal["fast", "deterministic", "compensated"]


class GaussianRasterExecutionPlan(StrictModule, NonTrainableState):
    """Static reference or tile-major Gaussian accumulation policy."""

    kind: GaussianRasterKind = eqx.field(static=True)
    tile_shape: tuple[int, int] = eqx.field(static=True)
    maximum_tile_routes: int | None = eqx.field(static=True)
    accumulation: GaussianRasterAccumulation = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: GaussianRasterKind = "reference",
        *,
        tile_shape: tuple[int, int] = (16, 16),
        maximum_tile_routes: int | None = None,
        accumulation: GaussianRasterAccumulation = "deterministic",
    ) -> None:
        if kind not in ("reference", "tiled"):
            raise ValueError("kind must be 'reference' or 'tiled'.")
        tiles = tuple(int(size) for size in tile_shape)
        if len(tiles) != 2 or any(size <= 0 for size in tiles):
            raise ValueError("tile_shape must contain two positive sizes.")
        capacity = None if maximum_tile_routes is None else int(maximum_tile_routes)
        if capacity is not None and capacity <= 0:
            raise ValueError("maximum_tile_routes must be positive when provided.")
        if accumulation not in ("fast", "deterministic", "compensated"):
            raise ValueError("Unknown Gaussian raster accumulation policy.")
        self.kind = kind
        self.tile_shape = tiles
        self.maximum_tile_routes = capacity
        self.accumulation = accumulation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gaussian-raster-execution-plan",
                "realization": kind,
                "tile_shape": list(tiles),
                "maximum_tile_routes": capacity,
                "accumulation": accumulation,
            }
        )


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
    route_count: Array
    route_capacity: Array
    route_overflow: Array
    status: Array


class GaussianRasterResult(StrictModule):
    """A particle image and per-particle rasterization evidence."""

    image: Array
    evidence: GaussianRasterEvidence
    successful: Array
    rasterizer_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)


class _GaussianRoutes(StrictModule):
    rows: Array
    columns: Array
    contributions: Array
    routed: Array
    supported: Array
    truncated: Array
    overflow: Array
    nonfinite: Array
    deposited_flux: Array
    status: Array


class GaussianRasterizer(StrictModule, NonTrainableState):
    """Bounded-memory Gaussian point-particle rasterizer.

    ``amplitude`` is integrated irradiance over the configured discrete support.
    Image coordinates are ``(row_down, column_right)``. Route indices are
    discrete; derivatives are piecewise derivatives for a fixed route topology.
    """

    execution: GaussianRasterExecutionPlan
    maximum_support_radius: int = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    rasterizer_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_support_radius: int = 6,
        *,
        cutoff: float = 3.5,
        execution: GaussianRasterExecutionPlan | None = None,
    ) -> None:
        radius = int(maximum_support_radius)
        cutoff_ = float(cutoff)
        execution_ = GaussianRasterExecutionPlan() if execution is None else execution
        if radius < 1:
            raise ValueError("maximum_support_radius must be positive.")
        if not isfinite(cutoff_) or cutoff_ <= 0.0:
            raise ValueError("cutoff must be finite and positive.")
        if not isinstance(execution_, GaussianRasterExecutionPlan):
            raise TypeError("execution must be GaussianRasterExecutionPlan.")
        self.execution = execution_
        self.maximum_support_radius = radius
        self.cutoff = cutoff_
        self.rasterizer_id = canonical_fingerprint(
            {
                "kind": "gaussian-particle-rasterizer",
                "maximum_support_radius": radius,
                "cutoff": cutoff_,
                "coordinate_convention": "row-down-column-right",
                "execution": execution_.plan_id,
            }
        )

    def _routes(
        self,
        geometry: ImagePlaneSupport,
        row_column: ArrayLike,
        amplitude: ArrayLike,
        sigma: ArrayLike,
        active: ArrayLike | None,
        /,
    ) -> tuple[_GaussianRoutes, Array, Array, Array]:
        if not isinstance(geometry, ImagePlaneSupport):
            raise TypeError("geometry must be ImagePlaneSupport.")
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
        radius = self.maximum_support_radius
        offsets = jnp.arange(-radius, radius + 1, dtype=jnp.int32)
        offset_rows, offset_columns = jnp.meshgrid(offsets, offsets, indexing="ij")
        offset_rows = offset_rows.reshape((-1,))
        offset_columns = offset_columns.reshape((-1,))
        finite = (
            jnp.all(jnp.isfinite(coordinates), axis=-1)
            & jnp.isfinite(amplitudes)
            & jnp.all(jnp.isfinite(sigmas), axis=-1)
            & (amplitudes >= 0.0)
            & jnp.all(sigmas > 0.0, axis=-1)
        )
        usable = active_ & finite
        safe_center = jnp.where(finite[:, None], coordinates, 0.0)
        safe_flux = jnp.where(finite, amplitudes, 0.0)
        safe_spread = jnp.where(finite[:, None], sigmas, 1.0)
        anchor = jnp.floor(safe_center).astype(jnp.int32)
        rows = anchor[:, 0, None] + offset_rows[None, :]
        columns = anchor[:, 1, None] + offset_columns[None, :]
        delta_row = rows.astype(dtype) - safe_center[:, 0, None]
        delta_column = columns.astype(dtype) - safe_center[:, 1, None]
        local_support = (jnp.abs(delta_row) <= self.cutoff * safe_spread[:, 0, None]) & (
            jnp.abs(delta_column) <= self.cutoff * safe_spread[:, 1, None]
        )
        inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
        squared_distance = (delta_row / safe_spread[:, 0, None]) ** 2 + (
            delta_column / safe_spread[:, 1, None]
        ) ** 2
        weights = jnp.exp(-0.5 * squared_distance) * local_support
        weight_sum = jnp.maximum(jnp.sum(weights, axis=-1), jnp.finfo(dtype).tiny)
        contributions = safe_flux[:, None] * weights / weight_sum[:, None]
        routed = usable[:, None] & local_support & inside
        contributions = jnp.where(routed, contributions, 0.0)
        supported = usable & jnp.any(local_support & inside, axis=-1)
        required_radius = jnp.ceil(self.cutoff * jnp.max(safe_spread, axis=-1)).astype(
            jnp.int32
        )
        overflow = usable & (required_radius > radius)
        truncated = overflow | (usable & jnp.any(local_support & ~inside, axis=-1))
        nonfinite = active_ & ~finite
        deposited = jnp.sum(contributions, axis=-1)
        status = jnp.where(
            ~active_,
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
        return (
            _GaussianRoutes(
                rows=rows,
                columns=columns,
                contributions=contributions,
                routed=routed,
                supported=supported,
                truncated=truncated,
                overflow=overflow,
                nonfinite=nonfinite,
                deposited_flux=deposited,
                status=status.astype(jnp.int32),
            ),
            active_,
            jnp.asarray(height, dtype=jnp.int32),
            jnp.asarray(width, dtype=jnp.int32),
        )

    def render(
        self,
        geometry: ImagePlaneSupport,
        row_column: ArrayLike,
        amplitude: ArrayLike,
        sigma: ArrayLike,
        active: ArrayLike | None = None,
        /,
    ) -> GaussianRasterResult:
        """Render one fixed-capacity particle set into ``geometry``."""
        routes, active_, height_array, width_array = self._routes(
            geometry, row_column, amplitude, sigma, active
        )
        height = int(geometry.image_shape[0])
        width = int(geometry.image_shape[1])
        del height_array, width_array
        route_count = jnp.sum(routes.routed, dtype=jnp.int32)
        candidate_capacity = int(routes.routed.size)
        declared_capacity = (
            candidate_capacity
            if self.execution.maximum_tile_routes is None
            else self.execution.maximum_tile_routes
        )
        route_overflow = route_count > declared_capacity
        usable_routes = routes.routed & ~route_overflow
        if self.execution.kind == "reference":
            image = self._render_reference(
                height,
                width,
                routes.contributions,
                routes.rows,
                routes.columns,
                usable_routes,
            )
        else:
            image = self._render_tiled(
                height,
                width,
                routes.contributions,
                routes.rows,
                routes.columns,
                usable_routes,
            )
        evidence = GaussianRasterEvidence(
            active=active_,
            supported=routes.supported,
            truncated=routes.truncated,
            overflow=routes.overflow,
            nonfinite=routes.nonfinite,
            deposited_flux=jnp.where(route_overflow, 0.0, routes.deposited_flux),
            active_count=jnp.sum(active_, dtype=jnp.int32),
            supported_count=jnp.sum(routes.supported, dtype=jnp.int32),
            truncated_count=jnp.sum(routes.truncated, dtype=jnp.int32),
            overflow_count=jnp.sum(routes.overflow, dtype=jnp.int32),
            route_count=route_count,
            route_capacity=jnp.asarray(declared_capacity, dtype=jnp.int32),
            route_overflow=route_overflow,
            status=routes.status,
        )
        return GaussianRasterResult(
            image=image,
            evidence=evidence,
            successful=~jnp.any(routes.nonfinite | routes.overflow) & ~route_overflow,
            rasterizer_id=self.rasterizer_id,
            geometry_id=geometry.geometry_id,
        )

    @staticmethod
    def _render_reference(
        height: int,
        width: int,
        contributions: Array,
        rows: Array,
        columns: Array,
        routed: Array,
        /,
    ) -> Array:
        dtype = contributions.dtype
        image = jnp.zeros((height, width), dtype=dtype)

        def deposit_one(current: Array, particle: tuple[Array, ...]):
            values, row, column, valid = particle
            indices = jnp.clip(row, 0, height - 1) * width + jnp.clip(
                column, 0, width - 1
            )
            payload = jnp.where(valid, values, 0.0)
            updated = current.reshape((-1,)).at[indices].add(payload)
            return updated.reshape((height, width)), None

        image, _ = jax.lax.scan(
            deposit_one,
            image,
            (contributions, rows, columns, routed),
        )
        return image

    def _render_tiled(
        self,
        height: int,
        width: int,
        contributions: Array,
        rows: Array,
        columns: Array,
        routed: Array,
        /,
    ) -> Array:
        tile_height, tile_width = self.execution.tile_shape
        tile_rows = (height + tile_height - 1) // tile_height
        tile_columns = (width + tile_width - 1) // tile_width
        tile_area = tile_height * tile_width
        safe_rows = jnp.clip(rows, 0, height - 1)
        safe_columns = jnp.clip(columns, 0, width - 1)
        tile_row = safe_rows // tile_height
        tile_column = safe_columns // tile_width
        local_row = safe_rows % tile_height
        local_column = safe_columns % tile_width
        tile_slots = (
            (tile_row * tile_columns + tile_column) * tile_area
            + local_row * tile_width
            + local_column
        ).reshape((-1,))
        route_values = contributions.reshape((-1,))
        route_valid = routed.reshape((-1,))
        route_capacity = int(tile_slots.size)
        relation = EdgeRelation(
            jnp.arange(route_capacity, dtype=jnp.int32),
            tile_slots,
            source_size=route_capacity,
            target_size=tile_rows * tile_columns * tile_area,
            valid=route_valid,
        )
        execution = RelationExecutionPlan().prepare(relation)
        tile_values, _ = execution.reduce(
            route_values,
            accumulation=self.execution.accumulation,
            output="dense",
        )
        image_rows, image_columns = jnp.meshgrid(
            jnp.arange(height, dtype=jnp.int32),
            jnp.arange(width, dtype=jnp.int32),
            indexing="ij",
        )
        image_tile_slots = (
            ((image_rows // tile_height) * tile_columns + image_columns // tile_width)
            * tile_area
            + (image_rows % tile_height) * tile_width
            + image_columns % tile_width
        )
        return tile_values[image_tile_slots]


def rasterize_gaussians(
    rasterizer: GaussianRasterizer,
    geometry: ImagePlaneSupport,
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
    "GaussianRasterAccumulation",
    "GaussianRasterEvidence",
    "GaussianRasterExecutionPlan",
    "GaussianRasterKind",
    "GaussianRasterResult",
    "GaussianRasterizer",
    "RASTER_CLIPPED",
    "RASTER_COMPLETE",
    "RASTER_INACTIVE",
    "RASTER_INVALID",
    "RASTER_SUPPORT_OVERFLOW",
    "rasterize_gaussians",
]
