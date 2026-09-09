#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class CompressibleLoadHistory(StrictModule, NonTrainableState):
    times: Array
    force_coefficients: Array
    moment_coefficients: Array
    shock_positions: Array
    valid: Array
    source_id: str = eqx.field(static=True)
    history_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        force_coefficients: ArrayLike,
        moment_coefficients: ArrayLike,
        shock_positions: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        source_id: str,
    ):
        times_ = jnp.asarray(times)
        force = jnp.asarray(force_coefficients)
        moment = jnp.asarray(moment_coefficients, dtype=force.dtype)
        shock = jnp.asarray(shock_positions, dtype=force.dtype)
        valid_ = (
            jnp.ones(times_.shape, dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        source = str(source_id)
        if (
            times_.ndim != 1
            or times_.size < 2
            or force.ndim != 2
            or force.shape[0] != times_.size
            or moment.shape[0] != times_.size
            or shock.shape != times_.shape
            or valid_.shape != times_.shape
            or not source
            or np.any(np.diff(np.asarray(times_)) <= 0.0)
        ):
            raise ValueError("Compressible load-history arrays are incompatible.")
        self.times = times_
        self.force_coefficients = force
        self.moment_coefficients = moment
        self.shock_positions = shock
        self.valid = valid_
        self.source_id = source
        self.history_id = canonical_fingerprint(
            {
                "kind": "compressible-load-history",
                "source": source,
                "capacity": int(times_.size),
                "force_dimension": int(force.shape[-1]),
            }
        )


class CompressibleShockTrackResult(StrictModule):
    location: Array
    face_index: Array
    pressure_gradient: Array
    peak_margin: Array
    valid: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class CompressibleShockTrackPlan(StrictModule, NonTrainableState):
    """Hard surface shock location with explicit search and confidence margin."""

    lower_coordinate: float = eqx.field(static=True)
    upper_coordinate: float = eqx.field(static=True)
    compression_sign: int = eqx.field(static=True)
    minimum_gradient: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinate_interval: tuple[float, float],
        /,
        *,
        compression_sign: Literal[-1, 1] = 1,
        minimum_gradient: float = 0.0,
    ):
        lower, upper = (float(value) for value in coordinate_interval)
        threshold = float(minimum_gradient)
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower >= upper
            or compression_sign not in (-1, 1)
            or not np.isfinite(threshold)
            or threshold < 0.0
        ):
            raise ValueError("Shock tracking interval, sign, or threshold is invalid.")
        self.lower_coordinate = lower
        self.upper_coordinate = upper
        self.compression_sign = compression_sign
        self.minimum_gradient = threshold
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compressible-shock-track",
                "interval": (lower, upper),
                "compression_sign": compression_sign,
                "minimum_gradient": threshold,
            }
        )

    def evaluate(
        self, surface_coordinate: ArrayLike, pressure: ArrayLike, /
    ) -> CompressibleShockTrackResult:
        coordinate = jnp.asarray(surface_coordinate)
        pressure_ = jnp.asarray(pressure)
        if (
            coordinate.ndim != 1
            or coordinate.size < 3
            or pressure_.shape[-1] != coordinate.size
            or np.any(np.diff(np.asarray(coordinate)) <= 0.0)
        ):
            raise ValueError("Shock track requires ordered surface pressure samples.")
        midpoint = 0.5 * (coordinate[:-1] + coordinate[1:])
        gradient = (pressure_[..., 1:] - pressure_[..., :-1]) / (
            coordinate[1:] - coordinate[:-1]
        )
        interval = (midpoint >= self.lower_coordinate) & (
            midpoint <= self.upper_coordinate
        )
        score = self.compression_sign * gradient
        score = jnp.where(interval, score, -jnp.inf)
        index = jnp.argmax(score, axis=-1)
        peak = jnp.take_along_axis(score, index[..., None], axis=-1)[..., 0]
        sorted_score = jnp.sort(score, axis=-1)
        second = sorted_score[..., -2]
        margin = peak - second
        location = jnp.take(midpoint, index)
        valid = (
            jnp.isfinite(peak)
            & jnp.isfinite(margin)
            & (peak >= self.minimum_gradient)
            & (margin >= 0.0)
        )
        successful = valid & jnp.all(jnp.isfinite(pressure_), axis=-1)
        return CompressibleShockTrackResult(
            location,
            index,
            peak,
            margin,
            valid,
            successful,
            self.plan_id,
        )


class CompressibleSnapshotMetricPlan(StrictModule, NonTrainableState):
    """Nondimensional square-root-volume coordinates for modal analysis."""

    cell_volumes: Array
    component_indices: tuple[int, ...] = eqx.field(static=True)
    component_scales: tuple[float, ...] = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_volumes: ArrayLike,
        component_indices: tuple[int, ...],
        component_scales: tuple[float, ...],
        /,
    ):
        volumes = jnp.asarray(cell_volumes)
        indices = tuple(int(index) for index in component_indices)
        scales = tuple(float(scale) for scale in component_scales)
        if (
            volumes.ndim == 0
            or np.any(~np.isfinite(np.asarray(volumes)))
            or np.any(np.asarray(volumes) <= 0.0)
            or not indices
            or len(set(indices)) != len(indices)
            or min(indices) < 0
            or len(scales) != len(indices)
            or any(not np.isfinite(scale) or scale <= 0.0 for scale in scales)
        ):
            raise ValueError("Snapshot volumes, components, or scales are invalid.")
        self.cell_volumes = volumes
        self.component_indices = indices
        self.component_scales = scales
        self.metric_id = canonical_fingerprint(
            {
                "kind": "compressible-snapshot-metric",
                "cell_volumes": array_tree_fingerprint(volumes),
                "component_indices": indices,
                "component_scales": scales,
            }
        )

    def encode(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        event_rank = self.cell_volumes.ndim + 1
        if (
            value.ndim < event_rank
            or value.shape[-event_rank:-1] != self.cell_volumes.shape
            or value.shape[-1] <= max(self.component_indices)
        ):
            raise ValueError("Snapshot state does not match metric cells or components.")
        leading_shape = value.shape[:-event_rank]
        selected = jnp.take(value, jnp.asarray(self.component_indices), axis=-1)
        scales = jnp.asarray(self.component_scales, dtype=value.dtype)
        weighted = selected / scales * jnp.sqrt(self.cell_volumes)[..., None]
        return weighted.reshape(
            leading_shape + (self.cell_volumes.size * len(self.component_indices),)
        )

    def decode(self, coordinates: ArrayLike, /) -> Array:
        value = jnp.asarray(coordinates)
        expected = self.cell_volumes.size * len(self.component_indices)
        if value.ndim < 1 or value.shape[-1] != expected:
            raise ValueError("Snapshot coordinates have the wrong metric dimension.")
        leading_shape = value.shape[:-1]
        weighted = value.reshape(
            leading_shape + self.cell_volumes.shape + (len(self.component_indices),)
        )
        scales = jnp.asarray(self.component_scales, dtype=value.dtype)
        return weighted / jnp.sqrt(self.cell_volumes)[..., None] * scales


__all__ = [
    "CompressibleLoadHistory",
    "CompressibleShockTrackPlan",
    "CompressibleShockTrackResult",
    "CompressibleSnapshotMetricPlan",
]
