#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ...astrodynamics import PreparedEarthOrientation, PreparedTimeRoute
from .._photometry import ObservationDataProvenance
from ._data import DetectorNetworkData
from ._status import GravitationalWaveStatus


_SPEED_OF_LIGHT_M_S = 299792458.0
SkyFrame = Literal["itrs", "gcrs"]


def _unit_vector(value: ArrayLike, role: str, /) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or np.any(~np.isfinite(vector)):
        raise ValueError(f"{role} must be a finite three-vector.")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{role} must be nonzero.")
    return vector / norm


class InterferometerGeometry(StrictModule, NonTrainableState):
    """Earth-fixed detector vertex and orthogonal arm directions."""

    vertex_m: Array
    x_arm: Array
    y_arm: Array
    detector_tensor: Array
    provenance: ObservationDataProvenance
    detector_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        detector_id: str,
        vertex_m: ArrayLike,
        x_arm: ArrayLike,
        y_arm: ArrayLike,
        provenance: ObservationDataProvenance,
        /,
        *,
        frame_id: str = "itrs",
    ):
        identifier = str(detector_id).strip()
        frame = str(frame_id).strip()
        vertex = np.asarray(vertex_m, dtype=float)
        x = _unit_vector(x_arm, "x_arm")
        y = _unit_vector(y_arm, "y_arm")
        if not identifier or not frame:
            raise ValueError("Detector and frame IDs must be non-empty.")
        if vertex.shape != (3,) or np.any(~np.isfinite(vertex)):
            raise ValueError("Detector vertex must be a finite three-vector.")
        if abs(float(np.dot(x, y))) > 1.0e-8:
            raise ValueError("Interferometer arms must be orthogonal.")
        if not isinstance(provenance, ObservationDataProvenance):
            raise TypeError("provenance must be ObservationDataProvenance.")
        tensor = 0.5 * (np.outer(x, x) - np.outer(y, y))
        self.vertex_m = jnp.asarray(vertex)
        self.x_arm = jnp.asarray(x)
        self.y_arm = jnp.asarray(y)
        self.detector_tensor = jnp.asarray(tensor)
        self.provenance = provenance
        self.detector_id = identifier
        self.frame_id = frame
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-interferometer-geometry",
                "detector": identifier,
                "frame": frame,
                "content": array_tree_fingerprint(
                    {"vertex_m": vertex, "x_arm": x, "y_arm": y}
                )["sha256"],
                "provenance": provenance.provenance_id,
            }
        )


class DetectorResponseResult(StrictModule):
    antenna: Array
    time_delay_seconds: Array
    source_direction: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class DetectorResponsePlan(StrictModule, NonTrainableState):
    """Prepared sky polarization, antenna, and geocentric delay response."""

    detector_tensors: Array
    vertices_m: Array
    earth_orientation: PreparedEarthOrientation | None
    gps_to_utc: PreparedTimeRoute | None
    detector_ids: tuple[str, ...] = eqx.field(static=True)
    geometry_ids: tuple[str, ...] = eqx.field(static=True)
    sky_frame: SkyFrame = eqx.field(static=True)
    time_origin_gps: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        network: DetectorNetworkData,
        geometries: Sequence[InterferometerGeometry],
        /,
        *,
        sky_frame: SkyFrame = "itrs",
        earth_orientation: PreparedEarthOrientation | None = None,
        gps_to_utc: PreparedTimeRoute | None = None,
        time_origin_gps: float | None = None,
    ):
        if not isinstance(network, DetectorNetworkData):
            raise TypeError("network must be DetectorNetworkData.")
        items = tuple(geometries)
        if tuple(item.detector_id for item in items) != network.detector_ids:
            raise ValueError("Detector geometry order must match network data exactly.")
        if any(item.frame_id.lower() != "itrs" for item in items):
            raise ValueError("Detector geometries must use the ITRS frame.")
        if sky_frame not in ("itrs", "gcrs"):
            raise ValueError("sky_frame must be 'itrs' or 'gcrs'.")
        if sky_frame == "gcrs" and (
            not isinstance(earth_orientation, PreparedEarthOrientation)
            or not isinstance(gps_to_utc, PreparedTimeRoute)
            or gps_to_utc.source_scale != "GPS"
            or gps_to_utc.target_scale != "UTC"
        ):
            raise ValueError(
                "GCRS response requires prepared GPS-to-UTC and Earth orientation."
            )
        if sky_frame == "itrs" and (
            earth_orientation is not None or gps_to_utc is not None
        ):
            raise ValueError("ITRS response does not consume Earth-orientation routes.")
        origin = (
            network.start_time_gps if time_origin_gps is None else float(time_origin_gps)
        )
        if not np.isfinite(origin):
            raise ValueError("time_origin_gps must be finite.")
        self.detector_tensors = jnp.stack(tuple(item.detector_tensor for item in items))
        self.vertices_m = jnp.stack(tuple(item.vertex_m for item in items))
        self.earth_orientation = earth_orientation
        self.gps_to_utc = gps_to_utc
        self.detector_ids = network.detector_ids
        self.geometry_ids = tuple(item.geometry_id for item in items)
        self.sky_frame = sky_frame
        self.time_origin_gps = origin
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-detector-response",
                "network": network.network_id,
                "geometries": list(self.geometry_ids),
                "sky_frame": sky_frame,
                "earth_orientation": None
                if earth_orientation is None
                else earth_orientation.plan_id,
                "time_route": None if gps_to_utc is None else gps_to_utc.route_id,
                "time_origin_gps": origin,
            }
        )

    def evaluate(
        self,
        right_ascension: ArrayLike,
        declination: ArrayLike,
        polarization: ArrayLike,
        geocent_time_gps: ArrayLike,
        /,
    ) -> DetectorResponseResult:
        ra = jnp.asarray(right_ascension).reshape(())
        dec = jnp.asarray(declination).reshape(())
        psi = jnp.asarray(polarization).reshape(())
        time = jnp.asarray(geocent_time_gps).reshape(())
        finite = jnp.all(jnp.isfinite(jnp.stack((ra, dec, psi, time))))
        domain = (dec >= -0.5 * jnp.pi) & (dec <= 0.5 * jnp.pi)
        cos_dec = jnp.cos(dec)
        direction = jnp.asarray(
            (cos_dec * jnp.cos(ra), cos_dec * jnp.sin(ra), jnp.sin(dec))
        )
        theta = jnp.asarray(
            (-jnp.sin(dec) * jnp.cos(ra), -jnp.sin(dec) * jnp.sin(ra), cos_dec)
        )
        phi = jnp.asarray((-jnp.sin(ra), jnp.cos(ra), 0.0))
        p = jnp.cos(psi) * theta + jnp.sin(psi) * phi
        q = -jnp.sin(psi) * theta + jnp.cos(psi) * phi
        plus = jnp.outer(p, p) - jnp.outer(q, q)
        cross = jnp.outer(p, q) + jnp.outer(q, p)
        orientation_valid = jnp.asarray(True)
        if self.sky_frame == "gcrs":
            route = self.gps_to_utc
            orientation = self.earth_orientation
            if route is None or orientation is None:
                raise RuntimeError(
                    "Prepared GCRS response lost its time/orientation route."
                )
            utc = route.apply(time - self.time_origin_gps)
            rotation = orientation.evaluate(utc.relative_seconds)
            matrix = rotation.rotation_gcrs_to_itrs
            direction = matrix @ direction
            plus = matrix @ plus @ matrix.T
            cross = matrix @ cross @ matrix.T
            orientation_valid = utc.valid & rotation.valid
        antenna = jnp.stack(
            (
                contract("dij,ij->d", self.detector_tensors, plus),
                contract("dij,ij->d", self.detector_tensors, cross),
            ),
            axis=-1,
        )
        delays = -contract("di,i->d", self.vertices_m, direction) / _SPEED_OF_LIGHT_M_S
        valid = finite & domain & orientation_valid & jnp.all(jnp.isfinite(antenna))
        status = jnp.where(
            valid,
            int(GravitationalWaveStatus.SUCCESS),
            int(GravitationalWaveStatus.INVALID_RESPONSE),
        ).astype(jnp.int32)
        return DetectorResponseResult(
            antenna, delays, direction, valid, status, self.plan_id
        )


__all__ = [
    "DetectorResponsePlan",
    "DetectorResponseResult",
    "InterferometerGeometry",
    "SkyFrame",
]
