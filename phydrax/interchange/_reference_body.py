#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units import conversion_factor, METER, RADIAN
from ._geospatial import GeospatialContract


class ReferenceBodyContract(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    gravitational_parameter_m3_s2: float = eqx.field(static=True)
    semi_major_axis_m: float = eqx.field(static=True)
    semi_minor_axis_m: float = eqx.field(static=True)
    rotation_rate_rad_s: float = eqx.field(static=True)
    reference_epoch_tai_s: float = eqx.field(static=True)
    gravity_model_id: str | None = eqx.field(static=True)
    magnetic_model_id: str | None = eqx.field(static=True)
    body_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        gravitational_parameter_m3_s2: float,
        semi_major_axis_m: float,
        semi_minor_axis_m: float,
        rotation_rate_rad_s: float,
        reference_epoch_tai_s: float,
        /,
        *,
        gravity_model_id: str | None = None,
        magnetic_model_id: str | None = None,
    ):
        name_ = str(name).strip()
        gm, major, minor, rotation, epoch = (
            float(gravitational_parameter_m3_s2),
            float(semi_major_axis_m),
            float(semi_minor_axis_m),
            float(rotation_rate_rad_s),
            float(reference_epoch_tai_s),
        )
        if (
            not name_
            or not np.isfinite(gm)
            or gm <= 0
            or not np.isfinite(major)
            or major <= 0
            or not np.isfinite(minor)
            or minor <= 0
            or minor > major
            or not np.isfinite(rotation)
            or not np.isfinite(epoch)
        ):
            raise ValueError(
                "Reference body identity, gravity, axes, rotation, or epoch is invalid."
            )
        gravity = None if gravity_model_id is None else str(gravity_model_id).strip()
        magnetic = None if magnetic_model_id is None else str(magnetic_model_id).strip()
        if gravity == "" or magnetic == "":
            raise ValueError("Reference field model IDs must be nonempty or None.")
        self.name = name_
        self.gravitational_parameter_m3_s2 = gm
        self.semi_major_axis_m, self.semi_minor_axis_m = major, minor
        self.rotation_rate_rad_s, self.reference_epoch_tai_s = rotation, epoch
        self.gravity_model_id, self.magnetic_model_id = gravity, magnetic
        self.body_id = canonical_fingerprint(
            {
                "kind": "reference-body",
                "name": name_,
                "gm_m3_s2": gm,
                "axes_m": (major, minor),
                "rotation_rate_rad_s": rotation,
                "reference_epoch_tai_s": epoch,
                "gravity_model_id": gravity,
                "magnetic_model_id": magnetic,
            }
        )

    @property
    def eccentricity_squared(self) -> float:
        return 1.0 - (self.semi_minor_axis_m / self.semi_major_axis_m) ** 2


class PlanetaryCoordinateContract(StrictModule, NonTrainableState):
    body: ReferenceBodyContract
    geospatial: GeospatialContract
    latitude_kind: Literal["planetocentric", "planetographic"] = eqx.field(static=True)
    longitude_positive: Literal["east", "west"] = eqx.field(static=True)
    frame: Literal["body-fixed", "inertial"] = eqx.field(static=True)
    longitude_to_radians: float = eqx.field(static=True)
    latitude_to_radians: float = eqx.field(static=True)
    height_to_meters: float = eqx.field(static=True)
    coordinate_id: str = eqx.field(static=True)

    def __init__(
        self,
        body: ReferenceBodyContract,
        geospatial: GeospatialContract,
        /,
        *,
        latitude_kind: Literal["planetocentric", "planetographic"],
        longitude_positive: Literal["east", "west"] = "east",
        frame: Literal["body-fixed", "inertial"] = "body-fixed",
    ):
        if not isinstance(body, ReferenceBodyContract) or not isinstance(
            geospatial, GeospatialContract
        ):
            raise TypeError(
                "Planetary coordinates require body and geospatial contracts."
            )
        if geospatial.horizontal_kind != "geographic" or geospatial.horizontal_axes != (
            "longitude",
            "latitude",
        ):
            raise ValueError(
                "Planetary angular coordinates require longitude/latitude geospatial axes."
            )
        longitude_unit, latitude_unit = geospatial.horizontal_units
        height_unit = geospatial.vertical_unit
        if longitude_unit is None or latitude_unit is None or height_unit is None:
            raise ValueError("Planetary angular and height units must be explicit.")
        longitude_factor = float(conversion_factor(longitude_unit, RADIAN))
        latitude_factor = float(conversion_factor(latitude_unit, RADIAN))
        height_factor = float(conversion_factor(height_unit, METER))
        if not all(
            np.isfinite(value)
            for value in (longitude_factor, latitude_factor, height_factor)
        ):
            raise ValueError("Planetary coordinate units cannot be normalized to SI.")
        if (
            latitude_kind not in ("planetocentric", "planetographic")
            or longitude_positive
            not in (
                "east",
                "west",
            )
            or frame not in ("body-fixed", "inertial")
        ):
            raise ValueError("Planetary latitude/longitude/frame convention is invalid.")
        self.body, self.geospatial = body, geospatial
        self.latitude_kind, self.longitude_positive, self.frame = (
            latitude_kind,
            longitude_positive,
            frame,
        )
        self.longitude_to_radians = float(longitude_factor)
        self.latitude_to_radians = float(latitude_factor)
        self.height_to_meters = float(height_factor)
        self.coordinate_id = canonical_fingerprint(
            {
                "kind": "planetary-coordinate-contract",
                "body": body.body_id,
                "geospatial": geospatial.coordinate_id,
                "latitude_kind": latitude_kind,
                "longitude_positive": longitude_positive,
                "frame": frame,
            }
        )

    def to_body_fixed_cartesian(self, longitude_latitude_height: ArrayLike, /) -> Array:
        if self.frame != "body-fixed":
            raise ValueError(
                "Angular-to-Cartesian conversion without a time is defined only in the body-fixed frame."
            )
        values = jnp.asarray(longitude_latitude_height)
        if values.ndim == 0 or values.shape[-1] != 3:
            raise ValueError(
                "Planetary angular coordinates need longitude/latitude/height."
            )
        longitude = (
            values[..., 0]
            * self.longitude_to_radians
            * (-1.0 if self.longitude_positive == "west" else 1.0)
        )
        latitude = values[..., 1] * self.latitude_to_radians
        height = values[..., 2] * self.height_to_meters
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)) | jnp.any(jnp.abs(latitude) > 0.5 * jnp.pi),
            "Planetary longitude/latitude/height must be finite and latitude physical.",
        )
        cosine = jnp.cos(latitude)
        if self.latitude_kind == "planetocentric":
            surface_radius = 1.0 / jnp.sqrt(
                cosine**2 / self.body.semi_major_axis_m**2
                + jnp.sin(latitude) ** 2 / self.body.semi_minor_axis_m**2
            )
            radius = surface_radius + height
            return jnp.stack(
                (
                    radius * cosine * jnp.cos(longitude),
                    radius * cosine * jnp.sin(longitude),
                    radius * jnp.sin(latitude),
                ),
                axis=-1,
            )
        eccentricity = self.body.eccentricity_squared
        normal = self.body.semi_major_axis_m / jnp.sqrt(
            1.0 - eccentricity * jnp.sin(latitude) ** 2
        )
        return jnp.stack(
            (
                (normal + height) * cosine * jnp.cos(longitude),
                (normal + height) * cosine * jnp.sin(longitude),
                (normal * (1.0 - eccentricity) + height) * jnp.sin(latitude),
            ),
            axis=-1,
        )

    def _rotation_inputs(
        self, positions_m: ArrayLike, tai_seconds: ArrayLike, /
    ) -> tuple[Array, Array]:
        positions = jnp.asarray(positions_m)
        time = jnp.asarray(tai_seconds)
        if positions.ndim == 0 or positions.shape[-1] != 3:
            raise ValueError("Planetary Cartesian positions need trailing XYZ.")
        batch_shape = np.broadcast_shapes(positions.shape[:-1], time.shape)
        positions = jnp.broadcast_to(positions, batch_shape + (3,))
        time = jnp.broadcast_to(time, batch_shape)
        positions = eqx.error_if(
            positions,
            jnp.any(~jnp.isfinite(positions)) | jnp.any(~jnp.isfinite(time)),
            "Planetary Cartesian positions and TAI times must be finite.",
        )
        return positions, time

    def body_fixed_to_inertial(
        self, positions_m: ArrayLike, tai_seconds: ArrayLike, /
    ) -> Array:
        positions, time = self._rotation_inputs(positions_m, tai_seconds)
        angle = self.body.rotation_rate_rad_s * (time - self.body.reference_epoch_tai_s)
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        x = cosine * positions[..., 0] - sine * positions[..., 1]
        y = sine * positions[..., 0] + cosine * positions[..., 1]
        return jnp.stack((x, y, positions[..., 2]), axis=-1)

    def inertial_to_body_fixed(
        self, positions_m: ArrayLike, tai_seconds: ArrayLike, /
    ) -> Array:
        positions, time = self._rotation_inputs(positions_m, tai_seconds)
        angle = -self.body.rotation_rate_rad_s * (time - self.body.reference_epoch_tai_s)
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        x = cosine * positions[..., 0] - sine * positions[..., 1]
        y = sine * positions[..., 0] + cosine * positions[..., 1]
        return jnp.stack((x, y, positions[..., 2]), axis=-1)


__all__ = ["PlanetaryCoordinateContract", "ReferenceBodyContract"]
