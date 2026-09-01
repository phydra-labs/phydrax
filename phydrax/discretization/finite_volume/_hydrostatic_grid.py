#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._structured import FiniteVolumeDiscretization


VerticalCoordinate = Literal["zstar", "partial-z"]
HorizontalCoordinate = Literal["cartesian", "latitude-longitude"]


def _face_neighbor_min(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    if periodic:
        faces = jnp.minimum(jnp.roll(moved, 1, axis=0), moved)
    else:
        interior = jnp.minimum(moved[:-1], moved[1:])
        faces = jnp.concatenate((moved[:1], interior, moved[-1:]), axis=0)
    return jnp.moveaxis(faces, 0, axis)


def _face_neighbor_average(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    if periodic:
        faces = 0.5 * (jnp.roll(moved, 1, axis=0) + moved)
    else:
        interior = 0.5 * (moved[:-1] + moved[1:])
        faces = jnp.concatenate((moved[:1], interior, moved[-1:]), axis=0)
    return jnp.moveaxis(faces, 0, axis)


def _face_difference(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    if periodic:
        difference = moved - jnp.roll(moved, 1, axis=0)
    else:
        interior = moved[1:] - moved[:-1]
        zero = jnp.zeros_like(moved[:1])
        difference = jnp.concatenate((zero, interior, zero), axis=0)
    return jnp.moveaxis(difference, 0, axis)


def _cell_net_flux(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    if periodic:
        difference = jnp.roll(moved, -1, axis=0) - moved
    else:
        difference = moved[1:] - moved[:-1]
    return jnp.moveaxis(difference, 0, axis)


class HydrostaticMetricEpoch(StrictModule):
    """One candidate hydrostatic volume/aperture/wet-support epoch."""

    eta: Array
    total_depth: Array
    layer_thickness: Array
    cell_volume: Array
    x_face_area: Array
    y_face_area: Array
    active_cell: Array
    active_x_face: Array
    active_y_face: Array
    wet_column: Array
    finite: Array
    valid: Array
    epoch_id: str = eqx.field(static=True)


class PreparedHydrostaticGrid(StrictModule, NonTrainableState):
    """Concrete tensor-z or bounded latitude-longitude hydrostatic geometry."""

    horizontal_coordinate: HorizontalCoordinate = eqx.field(static=True)
    vertical_coordinate: VerticalCoordinate = eqx.field(static=True)
    cell_shape: tuple[int, int, int] = eqx.field(static=True)
    horizontal_shape: tuple[int, int] = eqx.field(static=True)
    periodic: tuple[bool, bool] = eqx.field(static=True)
    cell_area: Array
    x_edge_length: Array
    y_edge_length: Array
    x_center_distance: Array
    y_center_distance: Array
    reference_vertical_faces: Array
    reference_layer_fraction: Array
    rest_depth: Array
    wet_depth: float = eqx.field(static=True)
    minimum_partial_fraction: float = eqx.field(static=True)
    longitude: Array
    latitude: Array
    coriolis: Array
    radius: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    @property
    def x_face_shape(self) -> tuple[int, int, int]:
        nx, ny, nz = self.cell_shape
        return (nx if self.periodic[0] else nx + 1, ny, nz)

    @property
    def y_face_shape(self) -> tuple[int, int, int]:
        nx, ny, nz = self.cell_shape
        return (nx, ny if self.periodic[1] else ny + 1, nz)

    def metric_epoch(self, eta: ArrayLike, /) -> HydrostaticMetricEpoch:
        eta_ = jnp.asarray(eta, dtype=self.cell_area.dtype)
        if eta_.shape != self.horizontal_shape:
            raise ValueError(f"Free surface must have shape {self.horizontal_shape}.")
        depth = jnp.maximum(self.rest_depth + eta_, 0.0)
        if self.vertical_coordinate == "zstar":
            thickness = depth[..., None] * self.reference_layer_fraction
        else:
            lower = self.reference_vertical_faces[:-1]
            upper = self.reference_vertical_faces[1:]
            bottom = -self.rest_depth
            wet_lower = jnp.maximum(lower, bottom[..., None])
            wet_upper = jnp.minimum(upper, eta_[..., None])
            thickness = jnp.maximum(wet_upper - wet_lower, 0.0)
            reference_width = upper - lower
            fractional = thickness / reference_width
            merge = (fractional > 0.0) & (fractional < self.minimum_partial_fraction)
            moved = jnp.moveaxis(thickness, -1, 0)
            merge_moved = jnp.moveaxis(merge, -1, 0)
            transferred = jnp.where(merge_moved[:-1], moved[:-1], 0.0)
            moved = moved.at[:-1].set(jnp.where(merge_moved[:-1], 0.0, moved[:-1]))
            moved = moved.at[1:].add(transferred)
            thickness = jnp.moveaxis(moved, 0, -1)
        active_cell = thickness > 0.0
        wet_column = depth > self.wet_depth
        active_cell = active_cell & wet_column[..., None]
        thickness = jnp.where(active_cell, thickness, 0.0)
        volume = self.cell_area[..., None] * thickness
        x_thickness = _face_neighbor_min(thickness, 0, self.periodic[0])
        y_thickness = _face_neighbor_min(thickness, 1, self.periodic[1])
        x_area = self.x_edge_length[..., None] * x_thickness
        y_area = self.y_edge_length[..., None] * y_thickness
        active_x = x_area > 0.0
        active_y = y_area > 0.0
        finite = (
            jnp.all(jnp.isfinite(depth))
            & jnp.all(jnp.isfinite(thickness))
            & jnp.all(jnp.isfinite(volume))
            & jnp.all(jnp.isfinite(x_area))
            & jnp.all(jnp.isfinite(y_area))
        )
        valid = (
            finite
            & jnp.all(volume >= 0.0)
            & jnp.all(x_area >= 0.0)
            & jnp.all(y_area >= 0.0)
            & jnp.all(
                jnp.abs(jnp.sum(thickness, axis=-1) - depth)
                <= 256.0 * jnp.finfo(depth.dtype).eps * jnp.maximum(depth, 1.0)
            )
        )
        return HydrostaticMetricEpoch(
            eta=eta_,
            total_depth=depth,
            layer_thickness=thickness,
            cell_volume=volume,
            x_face_area=x_area,
            y_face_area=y_area,
            active_cell=active_cell,
            active_x_face=active_x,
            active_y_face=active_y,
            wet_column=wet_column,
            finite=finite,
            valid=valid,
            epoch_id=canonical_fingerprint(
                {
                    "kind": "hydrostatic-metric-epoch",
                    "geometry": self.geometry_id,
                    "support": "eta-dependent",
                }
            ),
        )

    def net_cell_flux(self, fluxes: tuple[ArrayLike, ArrayLike], /) -> Array:
        x_flux = jnp.asarray(fluxes[0], dtype=self.cell_area.dtype)
        y_flux = jnp.asarray(fluxes[1], dtype=self.cell_area.dtype)
        if x_flux.shape != self.x_face_shape or y_flux.shape != self.y_face_shape:
            raise ValueError("Hydrostatic integrated face flux shapes are invalid.")
        return _cell_net_flux(x_flux, 0, self.periodic[0]) + _cell_net_flux(
            y_flux, 1, self.periodic[1]
        )

    def depth_integrate(
        self, transports: tuple[ArrayLike, ArrayLike], /
    ) -> tuple[Array, Array]:
        x = jnp.asarray(transports[0], dtype=self.cell_area.dtype)
        y = jnp.asarray(transports[1], dtype=self.cell_area.dtype)
        if x.shape != self.x_face_shape or y.shape != self.y_face_shape:
            raise ValueError("Hydrostatic transport shapes are invalid.")
        return jnp.sum(x, axis=-1), jnp.sum(y, axis=-1)

    def surface_net_transport(self, transports: tuple[ArrayLike, ArrayLike], /) -> Array:
        x, y = self.depth_integrate(transports)
        return _cell_net_flux(x, 0, self.periodic[0]) + _cell_net_flux(
            y, 1, self.periodic[1]
        )

    def surface_gradient(self, potential: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(potential, dtype=self.cell_area.dtype)
        if value.shape != self.horizontal_shape:
            raise ValueError("Surface potential shape is invalid.")
        x_difference = _face_difference(value, 0, self.periodic[0])
        y_difference = _face_difference(value, 1, self.periodic[1])
        return (
            x_difference / self.x_center_distance,
            y_difference / self.y_center_distance,
        )

    def layer_pressure_transport_force(
        self,
        potential: ArrayLike,
        epoch: HydrostaticMetricEpoch,
        /,
    ) -> tuple[Array, Array]:
        gx, gy = self.surface_gradient(potential)
        return (
            -epoch.x_face_area * gx[..., None],
            -epoch.y_face_area * gy[..., None],
        )

    def surface_net_flux(self, transports: tuple[ArrayLike, ArrayLike], /) -> Array:
        x = jnp.asarray(transports[0], dtype=self.cell_area.dtype)
        y = jnp.asarray(transports[1], dtype=self.cell_area.dtype)
        expected_x = self.x_face_shape[:-1]
        expected_y = self.y_face_shape[:-1]
        if x.shape != expected_x or y.shape != expected_y:
            raise ValueError("Barotropic transport shapes are invalid.")
        return _cell_net_flux(x, 0, self.periodic[0]) + _cell_net_flux(
            y, 1, self.periodic[1]
        )

    def diagnose_vertical_flux(self, transports: tuple[ArrayLike, ArrayLike], /) -> Array:
        x = jnp.asarray(transports[0], dtype=self.cell_area.dtype)
        y = jnp.asarray(transports[1], dtype=self.cell_area.dtype)
        layer_net = _cell_net_flux(x, 0, self.periodic[0]) + _cell_net_flux(
            y, 1, self.periodic[1]
        )
        bottom = jnp.zeros(self.horizontal_shape + (1,), dtype=layer_net.dtype)
        return jnp.concatenate((bottom, -jnp.cumsum(layer_net, axis=-1)), axis=-1)

    def face_average(self, value: ArrayLike, axis: int, /) -> Array:
        value_ = jnp.asarray(value, dtype=self.cell_area.dtype)
        return _face_neighbor_average(value_, axis, self.periodic[axis])

    def layer_gradient(self, potential: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(potential, dtype=self.cell_area.dtype)
        if value.shape != self.cell_shape:
            raise ValueError("Layer potential shape is invalid.")
        x_difference = _face_difference(value, 0, self.periodic[0])
        y_difference = _face_difference(value, 1, self.periodic[1])
        return (
            x_difference / self.x_center_distance[..., None],
            y_difference / self.y_center_distance[..., None],
        )

    def layer_potential_transport_force(
        self,
        potential: ArrayLike,
        epoch: HydrostaticMetricEpoch,
        /,
    ) -> tuple[Array, Array]:
        gx, gy = self.layer_gradient(potential)
        return -epoch.x_face_area * gx, -epoch.y_face_area * gy


class TensorZHydrostaticGridPlan(StrictModule, NonTrainableState):
    """Prepare Cartesian tensor-z hydrostatic metrics from an FV grid."""

    discretization: FiniteVolumeDiscretization
    rest_depth: Array
    vertical_coordinate: VerticalCoordinate = eqx.field(static=True)
    wet_depth: float = eqx.field(static=True)
    minimum_partial_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteVolumeDiscretization,
        rest_depth: ArrayLike,
        /,
        *,
        vertical_coordinate: VerticalCoordinate = "zstar",
        wet_depth: float = 1.0e-6,
        minimum_partial_fraction: float = 0.2,
    ):
        if not isinstance(discretization, FiniteVolumeDiscretization):
            raise TypeError("discretization must be FiniteVolumeDiscretization.")
        if len(discretization.cell_shape) != 3:
            raise ValueError("Hydrostatic tensor-z grids require three dimensions.")
        if discretization.grid.structured_axes[2].periodic:
            raise ValueError("Hydrostatic vertical axes must be bounded.")
        depth = jnp.asarray(rest_depth, dtype=discretization.cell_volumes.dtype)
        if depth.shape != discretization.cell_shape[:2]:
            raise ValueError("rest_depth must match the horizontal cell shape.")
        if bool(jnp.any(~jnp.isfinite(depth))) or bool(jnp.any(depth < 0.0)):
            raise ValueError("rest_depth must be finite and nonnegative.")
        if vertical_coordinate not in ("zstar", "partial-z"):
            raise ValueError("Unknown hydrostatic vertical-coordinate policy.")
        wet = float(wet_depth)
        fraction = float(minimum_partial_fraction)
        if not np.isfinite(wet) or wet < 0.0 or not 0.0 < fraction <= 1.0:
            raise ValueError("Invalid hydrostatic wet/partial-cell thresholds.")
        self.discretization = discretization
        self.rest_depth = depth
        self.vertical_coordinate = vertical_coordinate
        self.wet_depth = wet
        self.minimum_partial_fraction = fraction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "tensor-z-hydrostatic-grid-plan",
                "discretization": discretization.prepared_id,
                "vertical_coordinate": vertical_coordinate,
                "wet_depth": wet,
                "minimum_partial_fraction": fraction,
            }
        )

    def prepare(self) -> PreparedHydrostaticGrid:
        grid = self.discretization.grid
        x_axis, y_axis, z_axis = grid.structured_axes
        x_width = x_axis.interval_widths
        y_width = y_axis.interval_widths
        z_width = z_axis.interval_widths
        area = x_width[:, None] * y_width[None, :]
        nx, ny, nz = self.discretization.cell_shape
        x_count = nx if x_axis.periodic else nx + 1
        y_count = ny if y_axis.periodic else ny + 1
        x_edge = jnp.broadcast_to(y_width[None, :], (x_count, ny))
        y_edge = jnp.broadcast_to(x_width[:, None], (nx, y_count))
        x_distance = _face_neighbor_average(
            jnp.broadcast_to(x_width[:, None], (nx, ny)), 0, x_axis.periodic
        )
        y_distance = _face_neighbor_average(
            jnp.broadcast_to(y_width[None, :], (nx, ny)), 1, y_axis.periodic
        )
        vertical_faces = z_axis.point_coordinates
        fraction = z_width / jnp.sum(z_width)
        longitude = jnp.broadcast_to(x_axis.interval_centers[:, None], (nx, ny))
        latitude = jnp.broadcast_to(y_axis.interval_centers[None, :], (nx, ny))
        return PreparedHydrostaticGrid(
            horizontal_coordinate="cartesian",
            vertical_coordinate=self.vertical_coordinate,
            cell_shape=(nx, ny, nz),
            horizontal_shape=(nx, ny),
            periodic=(x_axis.periodic, y_axis.periodic),
            cell_area=area,
            x_edge_length=x_edge,
            y_edge_length=y_edge,
            x_center_distance=x_distance,
            y_center_distance=y_distance,
            reference_vertical_faces=vertical_faces,
            reference_layer_fraction=fraction,
            rest_depth=self.rest_depth,
            wet_depth=self.wet_depth,
            minimum_partial_fraction=self.minimum_partial_fraction,
            longitude=longitude,
            latitude=latitude,
            coriolis=jnp.zeros((nx, ny)),
            radius=math_inf(),
            geometry_id=canonical_fingerprint(
                {"kind": "prepared-tensor-z-hydrostatic-grid", "plan": self.plan_id}
            ),
        )


class LatitudeLongitudeHydrostaticGridPlan(StrictModule, NonTrainableState):
    """Prepare a bounded-away-from-poles spherical latitude-longitude C-grid."""

    longitude_faces: Array
    latitude_faces: Array
    vertical_faces: Array
    rest_depth: Array
    radius: float = eqx.field(static=True)
    rotation_rate: float = eqx.field(static=True)
    wet_depth: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        longitude_faces: ArrayLike,
        latitude_faces: ArrayLike,
        vertical_faces: ArrayLike,
        rest_depth: ArrayLike,
        /,
        *,
        radius: float = 6_371_000.0,
        rotation_rate: float = 7.292115e-5,
        wet_depth: float = 1.0e-6,
    ):
        lon = jnp.asarray(longitude_faces, dtype=float)
        lat = jnp.asarray(latitude_faces, dtype=float)
        z = jnp.asarray(vertical_faces, dtype=float)
        if lon.ndim != 1 or lat.ndim != 1 or z.ndim != 1:
            raise ValueError("Latitude-longitude faces must be one-dimensional.")
        if lon.size < 3 or lat.size < 3 or z.size < 3:
            raise ValueError("Latitude-longitude hydrostatic grids need >= 2 cells/axis.")
        if (
            bool(jnp.any(jnp.diff(lon) <= 0.0))
            or bool(jnp.any(jnp.diff(lat) <= 0.0))
            or bool(jnp.any(jnp.diff(z) <= 0.0))
        ):
            raise ValueError("Hydrostatic coordinate faces must increase strictly.")
        if float(lat[0]) <= -0.5 * np.pi or float(lat[-1]) >= 0.5 * np.pi:
            raise ValueError("Initial latitude-longitude grids must exclude both poles.")
        depth = jnp.asarray(rest_depth, dtype=float)
        expected = (lon.size - 1, lat.size - 1)
        if depth.shape != expected or bool(jnp.any(depth < 0.0)):
            raise ValueError(f"rest_depth must have shape {expected} and be nonnegative.")
        radius_ = float(radius)
        rotation_ = float(rotation_rate)
        wet = float(wet_depth)
        if not all(np.isfinite(v) for v in (radius_, rotation_, wet)) or radius_ <= 0:
            raise ValueError("Invalid spherical radius, rotation, or wet threshold.")
        self.longitude_faces = lon
        self.latitude_faces = lat
        self.vertical_faces = z
        self.rest_depth = depth
        self.radius = radius_
        self.rotation_rate = rotation_
        self.wet_depth = wet
        self.plan_id = canonical_fingerprint(
            {
                "kind": "latitude-longitude-hydrostatic-grid-plan",
                "radius": radius_,
                "rotation_rate": rotation_,
                "wet_depth": wet,
            }
        )

    def prepare(self) -> PreparedHydrostaticGrid:
        lon_f = self.longitude_faces
        lat_f = self.latitude_faces
        z_f = self.vertical_faces
        dlon = jnp.diff(lon_f)
        dlat = jnp.diff(lat_f)
        lon = 0.5 * (lon_f[:-1] + lon_f[1:])
        lat = 0.5 * (lat_f[:-1] + lat_f[1:])
        nx, ny = lon.size, lat.size
        area = (
            self.radius**2
            * dlon[:, None]
            * (jnp.sin(lat_f[1:]) - jnp.sin(lat_f[:-1]))[None, :]
        )
        x_edge = self.radius * jnp.broadcast_to(dlat[None, :], (nx, ny))
        y_edge = (
            self.radius
            * jnp.broadcast_to(dlon[:, None], (nx, ny + 1))
            * jnp.cos(lat_f)[None, :]
        )
        x_distance = (
            self.radius
            * jnp.cos(lat)[None, :]
            * jnp.broadcast_to(dlon[:, None], (nx, ny))
        )
        y_center_distance = self.radius * _face_neighbor_average(
            jnp.broadcast_to(dlat[None, :], (nx, ny)), 1, False
        )
        z_width = jnp.diff(z_f)
        return PreparedHydrostaticGrid(
            horizontal_coordinate="latitude-longitude",
            vertical_coordinate="zstar",
            cell_shape=(nx, ny, z_width.size),
            horizontal_shape=(nx, ny),
            periodic=(True, False),
            cell_area=area,
            x_edge_length=x_edge,
            y_edge_length=y_edge,
            x_center_distance=x_distance,
            y_center_distance=y_center_distance,
            reference_vertical_faces=z_f,
            reference_layer_fraction=z_width / jnp.sum(z_width),
            rest_depth=self.rest_depth,
            wet_depth=self.wet_depth,
            minimum_partial_fraction=0.2,
            longitude=jnp.broadcast_to(lon[:, None], (nx, ny)),
            latitude=jnp.broadcast_to(lat[None, :], (nx, ny)),
            coriolis=2.0
            * self.rotation_rate
            * jnp.sin(jnp.broadcast_to(lat[None, :], (nx, ny))),
            radius=self.radius,
            geometry_id=canonical_fingerprint(
                {
                    "kind": "prepared-latitude-longitude-hydrostatic-grid",
                    "plan": self.plan_id,
                }
            ),
        )


def math_inf() -> float:
    return float("inf")


__all__ = [
    "HydrostaticMetricEpoch",
    "LatitudeLongitudeHydrostaticGridPlan",
    "PreparedHydrostaticGrid",
    "TensorZHydrostaticGridPlan",
]
