#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-prepared Cartesian acoustic grids and exact multilinear acquisition maps."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._interpolation import apply_gather_stencil, GatherStencil, rectilinear_stencil
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....units import conversion_factor, METER, UnitDefinition


if TYPE_CHECKING:
    from ....interchange._geospatial import GeospatialContract


class AcousticGrid(StrictModule, NonTrainableState):
    """Static pressure-cell grid, normalized to metres at construction.

    ``origin`` is the centre of cell zero, not its lower corner. Two-dimensional
    grids describe translationally invariant, per-unit-thickness line-source
    acoustics, not a slice with three-dimensional geometric spreading.
    """

    shape: tuple[int, ...] = eqx.field(static=True)
    spacing: tuple[float, ...] = eqx.field(static=True)
    origin: tuple[float, ...] = eqx.field(static=True)
    coordinate_metadata: GeospatialContract | None = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)

    def __init__(
        self,
        shape: Sequence[int],
        spacing: Sequence[float],
        origin: Sequence[float] | None = None,
        /,
        *,
        length_unit: UnitDefinition = METER,
        coordinate_metadata: GeospatialContract | None = None,
    ):
        shape_ = tuple(int(size) for size in shape)
        if len(shape_) not in (2, 3) or any(size < 2 for size in shape_):
            raise ValueError(
                "Acoustic grids require two or three axes with at least two cells."
            )
        if any(int(size) != size for size in shape):
            raise ValueError("Acoustic grid shape must contain integers.")
        factor = conversion_factor(length_unit, METER)
        spacing_ = tuple(float(value) * factor for value in spacing)
        origin_ = (
            (0.0,) * len(shape_)
            if origin is None
            else tuple(float(value) * factor for value in origin)
        )
        if len(spacing_) != len(shape_) or len(origin_) != len(shape_):
            raise ValueError("Acoustic spacing and origin must match the grid dimension.")
        if not all(np.isfinite(value) and value > 0 for value in spacing_):
            raise ValueError("Acoustic cell spacing must be finite and positive.")
        if not all(np.isfinite(value) for value in origin_):
            raise ValueError("Acoustic origin must be finite.")
        if coordinate_metadata is not None:
            spatial = coordinate_metadata.require_cartesian(dimensions=len(shape_))
            if spatial.length_unit != length_unit:
                raise ValueError(
                    "Acoustic coordinate metadata must match the boundary length unit."
                )
        self.shape, self.spacing, self.origin = shape_, spacing_, origin_
        self.coordinate_metadata = coordinate_metadata
        self.grid_id = canonical_fingerprint(
            {
                "kind": "acoustic-cell-grid",
                "shape": shape_,
                "spacing_m": spacing_,
                "origin_m": origin_,
                "coordinates": None
                if coordinate_metadata is None
                else coordinate_metadata.geospatial_id,
            }
        )

    @property
    def dimensions(self) -> int:
        return len(self.shape)

    @property
    def cell_measure(self) -> float:
        """Cell volume in 3D; cell area per unit line length in 2D."""
        return prod(self.spacing)

    def axis_nodes(self) -> tuple[Array, ...]:
        return tuple(
            origin + spacing * jnp.arange(size, dtype=float)
            for size, spacing, origin in zip(
                self.shape, self.spacing, self.origin, strict=True
            )
        )


class PreparedAcousticSampling(StrictModule, NonTrainableState):
    """Fixed in-domain multilinear pressure sampling and its exact transpose.

    Preparation rejects extrapolation. ``transpose`` returns grid coefficients;
    source injection separately divides those coefficients by cell measure.
    Coordinates are not differentiable after host preparation.
    """

    stencil: GatherStencil
    positions: Array
    grid_shape: tuple[int, ...] = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)
    sampling_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: AcousticGrid,
        positions: ArrayLike,
        /,
        *,
        length_unit: UnitDefinition = METER,
    ):
        points = np.asarray(positions, dtype=float) * float(
            conversion_factor(length_unit, METER)
        )
        if points.ndim != 2 or points.shape[1] != grid.dimensions or points.shape[0] == 0:
            raise ValueError("Acoustic positions require shape (count, grid_dimensions).")
        lower = np.asarray(grid.origin)
        upper = lower + np.asarray(grid.spacing) * (np.asarray(grid.shape) - 1)
        if (
            not np.all(np.isfinite(points))
            or np.any(points < lower)
            or np.any(points > upper)
        ):
            raise ValueError(
                "Acoustic sample positions must lie within pressure-cell centres."
            )
        self.positions = jnp.asarray(points)
        self.stencil = rectilinear_stencil(
            grid.axis_nodes(), self.positions, boundary=("constant",) * grid.dimensions
        )
        self.grid_shape, self.grid_id = grid.shape, grid.grid_id
        self.sampling_id = canonical_fingerprint(
            {
                "kind": "acoustic-sampling",
                "grid": grid.grid_id,
                "positions_m": points.tolist(),
            }
        )

    @property
    def count(self) -> int:
        return self.positions.shape[0]

    def apply(self, pressure: ArrayLike, /) -> Array:
        values = jnp.asarray(pressure)
        if values.shape != self.grid_shape:
            raise ValueError("Acoustic pressure must match the prepared grid shape.")
        return apply_gather_stencil(values.reshape((-1,)), self.stencil).values

    def transpose(self, values: ArrayLike, /) -> Array:
        values_ = jnp.asarray(values, dtype=float)
        if values_.shape != (self.count,):
            raise ValueError("Acoustic transpose requires one coefficient per sample.")
        return jax.linear_transpose(
            self.apply, jnp.zeros(self.grid_shape, dtype=values_.dtype)
        )(values_)[0]


class SeismicAcquisition(StrictModule, NonTrainableState):
    """Prepared simultaneous monopole sources and pressure receivers."""

    sources: PreparedAcousticSampling
    receivers: PreparedAcousticSampling
    acquisition_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: AcousticGrid,
        source_positions: ArrayLike,
        receiver_positions: ArrayLike,
        /,
        *,
        length_unit: UnitDefinition = METER,
    ):
        self.sources = PreparedAcousticSampling(
            grid, source_positions, length_unit=length_unit
        )
        self.receivers = PreparedAcousticSampling(
            grid, receiver_positions, length_unit=length_unit
        )
        self.acquisition_id = canonical_fingerprint(
            {
                "kind": "seismic-acquisition",
                "sources": self.sources.sampling_id,
                "receivers": self.receivers.sampling_id,
            }
        )


__all__ = ["AcousticGrid", "PreparedAcousticSampling", "SeismicAcquisition"]
