#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import GatherStencil
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._gr_medium import (
    _runtime_sampling_stencil,
    _sample_medium_fields,
    FastLightSnapshot,
    GRMediumFieldUnits,
    GRMediumSample,
)


if TYPE_CHECKING:
    from ._gr_transfer import PolarizedRayPath


class FixedGRWorldtubeSamplingPlan(StrictModule, NonTrainableState):
    """Prepared 16-corner spacetime gather with fixed event capacity."""

    stencil: GatherStencil
    interpolation_smooth: Array
    event_shape: tuple[int, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        stencil: GatherStencil,
        interpolation_smooth: ArrayLike,
        /,
        *,
        event_shape: Sequence[int],
        source_id: str,
        event_fingerprint: str,
    ):
        shape = tuple(int(value) for value in event_shape)
        smooth = jax.lax.stop_gradient(jnp.asarray(interpolation_smooth, dtype=bool))
        if not isinstance(stencil, GatherStencil):
            raise TypeError("stencil must be a GatherStencil.")
        if stencil.support.shape != shape or smooth.shape != shape:
            raise ValueError("Worldtube stencil and event shapes must match.")
        identifier = str(source_id).strip()
        event_id = str(event_fingerprint).strip()
        if not identifier or not event_id:
            raise ValueError("Worldtube source and event identities must be non-empty.")
        self.stencil = stencil
        self.interpolation_smooth = smooth
        self.event_shape = shape
        self.source_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-gr-worldtube-sampling",
                "source": identifier,
                "events": event_id,
                "event_shape": shape,
                "stencil_capacity": int(stencil.indices.shape[-1]),
            }
        )


class MonotoneSlowLightWorldtube(StrictModule, NonTrainableState):
    """Strictly time-ordered GR plasma snapshots with spacetime interpolation.

    Events are supplied as ``(coordinate_time, x1, x2, x3)``.  Interpolation is
    multilinear over two time slices and eight spatial corners.  Queries outside
    the closed sampled worldtube, or touching any invalid source node, return
    explicit failed support evidence; there is no temporal extrapolation.
    """

    coordinate_times: Array
    coordinate_axes: tuple[Array, Array, Array]
    rest_mass_density: Array
    electron_number_density: Array
    electron_temperature: Array
    fluid_four_velocity: Array
    magnetic_four_vector: Array
    source_mask: Array
    units: GRMediumFieldUnits
    snapshot_ids: tuple[str, ...] = eqx.field(static=True)
    time_count: int = eqx.field(static=True)
    spatial_shape: tuple[int, int, int] = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)
    worldtube_id: str = eqx.field(static=True)

    def __init__(self, snapshots: Sequence[FastLightSnapshot], /, *, worldtube_id: str):
        values = tuple(snapshots)
        identifier = str(worldtube_id).strip()
        if (
            len(values) < 2
            or any(not isinstance(value, FastLightSnapshot) for value in values)
            or not identifier
        ):
            raise ValueError(
                "A slow-light worldtube requires at least two snapshots and an ID."
            )
        reference = values[0]
        times_host = np.asarray(
            [float(np.asarray(value.coordinate_time)) for value in values], dtype=float
        )
        if np.any(~np.isfinite(times_host)) or np.any(np.diff(times_host) <= 0.0):
            raise ValueError(
                "Slow-light snapshot times must be finite and strictly increasing."
            )
        if any(
            value.spatial_shape != reference.spatial_shape
            or value.units.units_id != reference.units.units_id
            or value.convention.convention_id != reference.convention.convention_id
            or value.chart_id != reference.chart_id
            or any(
                not np.array_equal(np.asarray(axis), np.asarray(reference_axis))
                for axis, reference_axis in zip(
                    value.coordinate_axes, reference.coordinate_axes, strict=True
                )
            )
            for value in values[1:]
        ):
            raise ValueError(
                "Slow-light snapshots must share grid, units, convention, and chart."
            )
        self.coordinate_times = jax.lax.stop_gradient(jnp.asarray(times_host))
        self.coordinate_axes = reference.coordinate_axes
        self.rest_mass_density = jax.lax.stop_gradient(
            jnp.stack(tuple(value.rest_mass_density for value in values), axis=0)
        )
        self.electron_number_density = jax.lax.stop_gradient(
            jnp.stack(tuple(value.electron_number_density for value in values), axis=0)
        )
        self.electron_temperature = jax.lax.stop_gradient(
            jnp.stack(tuple(value.electron_temperature for value in values), axis=0)
        )
        self.fluid_four_velocity = jax.lax.stop_gradient(
            jnp.stack(tuple(value.fluid_four_velocity for value in values), axis=0)
        )
        self.magnetic_four_vector = jax.lax.stop_gradient(
            jnp.stack(tuple(value.magnetic_four_vector for value in values), axis=0)
        )
        self.source_mask = jax.lax.stop_gradient(
            jnp.stack(tuple(value.source_mask for value in values), axis=0)
        )
        self.units = reference.units
        self.snapshot_ids = tuple(value.snapshot_id for value in values)
        self.time_count = len(values)
        self.spatial_shape = reference.spatial_shape
        self.convention_id = reference.convention.convention_id
        self.chart_id = reference.chart_id
        self.worldtube_id = canonical_fingerprint(
            {
                "kind": "monotone-slow-light-worldtube",
                "worldtube_id": identifier,
                "snapshots": self.snapshot_ids,
                "times": times_host.tolist(),
                "units": reference.units.units_id,
                "convention": reference.convention.convention_id,
                "chart": reference.chart_id,
            }
        )

    def prepare_sampling(
        self, event_coordinates: ArrayLike, /
    ) -> FixedGRWorldtubeSamplingPlan:
        events_host = np.asarray(event_coordinates, dtype=float)
        if events_host.ndim < 1 or events_host.shape[-1] != 4:
            raise ValueError(
                "Slow-light event coordinates must end in (time, x1, x2, x3)."
            )
        axes = (self.coordinate_times,) + self.coordinate_axes
        stencil, smooth = _runtime_sampling_stencil(axes, jnp.asarray(events_host))
        return FixedGRWorldtubeSamplingPlan(
            stencil,
            smooth,
            event_shape=events_host.shape[:-1],
            source_id=self.worldtube_id,
            event_fingerprint=array_tree_fingerprint(events_host),
        )

    def prepare_path_sampling(
        self, path: PolarizedRayPath, /
    ) -> FixedGRWorldtubeSamplingPlan:
        """Prepare active spacetime-midpoint sampling for one exact GR ray path."""

        from ._gr_transfer import PolarizedRayPath

        if not isinstance(path, PolarizedRayPath):
            raise TypeError("path must be a PolarizedRayPath.")
        if (
            self.chart_id != path.chart_id
            or self.convention_id != path.convention_id
            or self.units.scale.scale_id != path.scale_id
            or self.units.coordinate_unit.unit_id != path.coordinate_unit_id
        ):
            raise ValueError(
                "Slow-light worldtube and GR ray path must share exact chart, "
                "relativity convention, scale, and coordinate unit identities."
            )
        midpoints = 0.5 * (path.coordinates[:-1] + path.coordinates[1:])
        prepared = self.prepare_sampling(midpoints)
        segment_support = (
            path.active[:-1] & path.active[1:] & path.valid[:-1] & path.valid[1:]
        )
        stencil = GatherStencil(
            indices=prepared.stencil.indices,
            weights=prepared.stencil.weights,
            source_size=prepared.stencil.source_size,
            valid=prepared.stencil.valid,
            support=prepared.stencil.support & segment_support,
        )
        return FixedGRWorldtubeSamplingPlan(
            stencil,
            prepared.interpolation_smooth & segment_support,
            event_shape=midpoints.shape[:-1],
            source_id=self.worldtube_id,
            event_fingerprint=canonical_fingerprint(
                {
                    "kind": "slow-light-ray-path-midpoints",
                    "path": path.path_id,
                    "worldtube": self.worldtube_id,
                }
            ),
        )

    def sample(self, plan: FixedGRWorldtubeSamplingPlan, /) -> GRMediumSample:
        if not isinstance(plan, FixedGRWorldtubeSamplingPlan):
            raise TypeError("plan must be a FixedGRWorldtubeSamplingPlan.")
        if plan.source_id != self.worldtube_id:
            raise ValueError("Sampling plan was prepared for a different worldtube.")
        return _sample_medium_fields(
            plan.stencil,
            plan.interpolation_smooth,
            plan.source_id,
            self.rest_mass_density,
            self.electron_number_density,
            self.electron_temperature,
            self.fluid_four_velocity,
            self.magnetic_four_vector,
            self.source_mask,
            units_id=self.units.units_id,
            convention_id=self.convention_id,
            chart_id=self.chart_id,
        )

    def evaluate(self, event_coordinates: ArrayLike, /) -> GRMediumSample:
        """Sample dynamic events through a fixed 16-corner JAX stencil."""

        events = jnp.asarray(event_coordinates)
        if events.ndim < 1 or events.shape[-1] != 4:
            raise ValueError(
                "Slow-light event coordinates must end in (time, x1, x2, x3)."
            )
        axes = (self.coordinate_times,) + self.coordinate_axes
        stencil, smooth = _runtime_sampling_stencil(axes, events)
        return _sample_medium_fields(
            stencil,
            smooth,
            self.worldtube_id,
            self.rest_mass_density,
            self.electron_number_density,
            self.electron_temperature,
            self.fluid_four_velocity,
            self.magnetic_four_vector,
            self.source_mask,
            units_id=self.units.units_id,
            convention_id=self.convention_id,
            chart_id=self.chart_id,
        )
