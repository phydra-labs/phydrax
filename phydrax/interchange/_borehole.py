#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units import conversion_factor, METER, UnitDefinition
from ._geospatial import GeospatialContract
from ._resource import ResourceManifest


class BoreholeInterval(StrictModule, NonTrainableState):
    start_measured_depth_m: float = eqx.field(static=True)
    end_measured_depth_m: float = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    inner_diameter_m: float | None = eqx.field(static=True)
    outer_diameter_m: float | None = eqx.field(static=True)
    interval_id: str = eqx.field(static=True)

    def __init__(
        self,
        start_measured_depth: float,
        end_measured_depth: float,
        kind: str,
        material_id: str,
        /,
        *,
        length_unit: UnitDefinition = METER,
        inner_diameter: float | None = None,
        outer_diameter: float | None = None,
    ):
        factor = float(conversion_factor(length_unit, METER))
        start = float(start_measured_depth) * factor
        end = float(end_measured_depth) * factor
        kind_, material = str(kind).strip(), str(material_id).strip()
        inner = None if inner_diameter is None else float(inner_diameter) * factor
        outer = None if outer_diameter is None else float(outer_diameter) * factor
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            raise ValueError(
                "Borehole interval depths must be finite, nonnegative, and ordered."
            )
        if not kind_ or not material:
            raise ValueError("Borehole interval kind and material ID are required.")
        if inner is not None and (not np.isfinite(inner) or inner <= 0):
            raise ValueError("Inner diameter must be finite and positive or None.")
        if outer is not None and (not np.isfinite(outer) or outer <= 0):
            raise ValueError("Outer diameter must be finite and positive or None.")
        if inner is not None and outer is not None and inner >= outer:
            raise ValueError(
                "Borehole inner diameter must be smaller than outer diameter."
            )
        self.start_measured_depth_m, self.end_measured_depth_m = start, end
        self.kind, self.material_id = kind_, material
        self.inner_diameter_m, self.outer_diameter_m = inner, outer
        self.interval_id = canonical_fingerprint(
            {
                "kind": "borehole-interval",
                "depths_m": (start, end),
                "interval_kind": kind_,
                "material_id": material,
                "diameters_m": (inner, outer),
            }
        )


class PreparedBoreholeSampling(StrictModule, NonTrainableState):
    trajectory_id: str = eqx.field(static=True)
    measured_depth_m: Array
    lower_indices: Array
    weights: Array

    def apply(self, station_values: ArrayLike) -> Array:
        values = jnp.asarray(station_values)
        station_count = int(jnp.max(self.lower_indices)) + 2
        if values.ndim == 0 or values.shape[0] < station_count:
            raise ValueError(
                "Station values do not cover the prepared borehole segments."
            )
        lower = values[self.lower_indices]
        upper = values[self.lower_indices + 1]
        weight = self.weights.reshape(self.weights.shape + (1,) * (values.ndim - 1))
        return (1.0 - weight) * lower + weight * upper

    def transpose(self, sampled_values: ArrayLike, station_count: int) -> Array:
        values = jnp.asarray(sampled_values)
        if values.ndim == 0 or values.shape[0] != self.weights.size:
            raise ValueError("Sample values do not match the prepared borehole query.")
        count = int(station_count)
        if count < 2 or jnp.max(self.lower_indices) + 1 >= count:
            raise ValueError(
                "Station count does not cover the prepared trajectory segments."
            )
        weight = self.weights.reshape(self.weights.shape + (1,) * (values.ndim - 1))
        result = jnp.zeros((count,) + values.shape[1:], dtype=values.dtype)
        result = result.at[self.lower_indices].add((1.0 - weight) * values)
        return result.at[self.lower_indices + 1].add(weight * values)


class BoreholeTrajectory(StrictModule, NonTrainableState):
    """Piecewise-linear measured-depth trajectory in a qualified Cartesian frame.

    XYZ stations and measured depth are authoritative. Optional tool orientations
    are proper rotation matrices whose rows are tool axes expressed in the
    geospatial frame. Segment changes are nondifferentiable preparation events.
    """

    name: str = eqx.field(static=True)
    measured_depth_m: Array
    positions_m: Array
    tool_orientations: Array | None
    coordinates: GeospatialContract
    intervals: tuple[BoreholeInterval, ...]
    resources: tuple[ResourceManifest, ...] = eqx.field(static=True)
    trajectory_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        measured_depth: ArrayLike,
        positions: ArrayLike,
        coordinates: GeospatialContract,
        /,
        *,
        length_unit: UnitDefinition = METER,
        tool_orientations: ArrayLike | None = None,
        intervals: Sequence[BoreholeInterval] = (),
        resources: Sequence[ResourceManifest] = (),
        geometry_tolerance: float = 1e-10,
    ):
        name_ = str(name).strip()
        if not name_:
            raise ValueError("Borehole name is required.")
        if not isinstance(coordinates, GeospatialContract):
            raise TypeError("Borehole coordinates require GeospatialContract.")
        spatial = coordinates.require_cartesian(dimensions=3)
        if spatial.length_unit != length_unit:
            raise ValueError("Borehole array unit and geospatial length unit disagree.")
        factor = float(conversion_factor(length_unit, METER))
        depth = np.asarray(measured_depth, dtype=float) * factor
        xyz = np.asarray(positions, dtype=float) * factor
        tolerance = float(geometry_tolerance)
        if depth.ndim != 1 or depth.size < 2 or xyz.shape != (depth.size, 3):
            raise ValueError(
                "Borehole trajectory requires matching depth and XYZ stations."
            )
        if (
            np.any(~np.isfinite(depth))
            or np.any(~np.isfinite(xyz))
            or np.any(np.diff(depth) <= 0)
            or not np.isfinite(tolerance)
            or tolerance <= 0
        ):
            raise ValueError(
                "Borehole stations and tolerance must be finite and ordered."
            )
        chord = np.sqrt(np.sum(np.diff(xyz, axis=0) ** 2, axis=1))
        increments = np.diff(depth)
        if np.any(chord > increments * (1.0 + tolerance)):
            raise ValueError(
                "Measured-depth increment cannot be shorter than station chord."
            )
        orientation = None
        if tool_orientations is not None:
            orientation = np.asarray(tool_orientations, dtype=float)
            if orientation.shape != (depth.size, 3, 3) or np.any(
                ~np.isfinite(orientation)
            ):
                raise ValueError(
                    "Tool orientations must be one finite 3x3 matrix per station."
                )
            gram = np.einsum("...ji,...jk->...ik", orientation, orientation)
            determinant = np.linalg.det(orientation)
            if not np.allclose(
                gram, np.eye(3), rtol=tolerance, atol=tolerance
            ) or not np.allclose(determinant, 1.0, rtol=tolerance, atol=tolerance):
                raise ValueError(
                    "Tool orientations must be proper orthonormal rotations."
                )
        intervals_ = tuple(intervals)
        if any(not isinstance(item, BoreholeInterval) for item in intervals_):
            raise TypeError("Borehole intervals must contain BoreholeInterval values.")
        for interval in intervals_:
            if (
                interval.start_measured_depth_m < depth[0]
                or interval.end_measured_depth_m > depth[-1]
            ):
                raise ValueError(
                    "Borehole interval lies outside trajectory measured depth."
                )
        resources_ = tuple(resources)
        if any(not isinstance(item, ResourceManifest) for item in resources_):
            raise TypeError("Borehole resources must contain ResourceManifest values.")
        self.name = name_
        self.measured_depth_m, self.positions_m = jnp.asarray(depth), jnp.asarray(xyz)
        self.tool_orientations = None if orientation is None else jnp.asarray(orientation)
        self.coordinates = coordinates
        self.intervals = tuple(
            sorted(intervals_, key=lambda item: item.start_measured_depth_m)
        )
        self.resources = resources_
        self.trajectory_id = canonical_fingerprint(
            {
                "kind": "borehole-trajectory",
                "name": name_,
                "measured_depth_m": depth,
                "positions_m": xyz,
                "tool_orientations": orientation,
                "coordinate_id": coordinates.coordinate_id,
                "intervals": [item.interval_id for item in self.intervals],
                "resources": [item.manifest_id for item in resources_],
            }
        )

    def prepare_sampling(
        self, measured_depth: ArrayLike, /, *, length_unit: UnitDefinition = METER
    ) -> PreparedBoreholeSampling:
        queries = np.asarray(measured_depth, dtype=float) * float(
            conversion_factor(length_unit, METER)
        )
        stations = np.asarray(self.measured_depth_m)
        if queries.ndim != 1 or np.any(~np.isfinite(queries)):
            raise ValueError("Borehole query depths must be a finite vector.")
        if np.any((queries < stations[0]) | (queries > stations[-1])):
            raise ValueError("Borehole query lies outside the trajectory.")
        lower = np.searchsorted(stations, queries, side="right") - 1
        lower = np.minimum(np.maximum(lower, 0), stations.size - 2)
        weights = (queries - stations[lower]) / (stations[lower + 1] - stations[lower])
        return PreparedBoreholeSampling(
            self.trajectory_id,
            jnp.asarray(queries),
            jnp.asarray(lower, dtype=jnp.int32),
            jnp.asarray(weights),
        )

    def sample_positions(
        self, measured_depth: ArrayLike, /, *, length_unit: UnitDefinition = METER
    ) -> Array:
        return self.prepare_sampling(measured_depth, length_unit=length_unit).apply(
            self.positions_m
        )


__all__ = ["BoreholeInterval", "BoreholeTrajectory", "PreparedBoreholeSampling"]
