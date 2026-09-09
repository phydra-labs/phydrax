#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared nested star-shaped flux surfaces and conservative radial measures."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume import MetricLinePlan, PreparedMetricLine
from ._equilibrium import AxisymmetricEquilibrium


class FluxSurfaceEvidence(StrictModule):
    minimum_cell_volume_m3: Array
    maximum_contour_gap_m: Array
    topology_fixed: Array
    finite: Array
    successful: Array


class PreparedFluxSurfaceGeometry(StrictModule, NonTrainableState):
    rho_faces: Array
    rho_cells: Array
    contours_rz_m: Array
    enclosed_volume_m3: Array
    cell_volume_m3: Array
    surface_area_m2: Array
    major_radius_m: Array
    minor_radius_m: Array
    safety_factor: Array
    metric_line: PreparedMetricLine
    evidence: FluxSurfaceEvidence
    equilibrium_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    differentiation: str = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class FluxSurfaceGeometry:
    rho_faces: np.ndarray
    contours_rz_m: np.ndarray
    enclosed_volume_m3: np.ndarray
    surface_area_m2: np.ndarray
    major_radius_m: np.ndarray
    minor_radius_m: np.ndarray
    safety_factor: np.ndarray
    equilibrium_id: str
    geometry_id: str = field(init=False)

    def __post_init__(self) -> None:
        rho = np.array(self.rho_faces, dtype=np.float64, copy=True)
        contours = np.array(self.contours_rz_m, dtype=np.float64, copy=True)
        count = rho.size
        arrays = tuple(
            np.array(value, dtype=np.float64, copy=True)
            for value in (
                self.enclosed_volume_m3,
                self.surface_area_m2,
                self.major_radius_m,
                self.minor_radius_m,
                self.safety_factor,
            )
        )
        if rho.ndim != 1 or count < 2 or rho[0] != 0.0 or np.any(np.diff(rho) <= 0.0):
            raise ValueError("rho_faces must start at zero and increase strictly.")
        if contours.ndim != 3 or contours.shape[0] != count or contours.shape[2] != 2:
            raise ValueError("contours_rz_m must have shape (face, poloidal, 2).")
        if any(value.shape != (count,) for value in arrays):
            raise ValueError("Flux-surface metric arrays must match rho_faces.")
        if np.any(~np.isfinite(contours)) or any(np.any(~np.isfinite(v)) for v in arrays):
            raise ValueError("Flux-surface geometry must be finite.")
        volume, area, major, minor, safety = arrays
        if volume[0] != 0.0 or area[0] != 0.0 or minor[0] != 0.0:
            raise ValueError("Magnetic-axis surface must have zero measure and radius.")
        if np.any(np.diff(volume) <= 0.0) or np.any(area[1:] <= 0.0):
            raise ValueError("Nested surfaces require positive volumes and areas.")
        equilibrium = str(self.equilibrium_id).strip()
        if not equilibrium or equilibrium != self.equilibrium_id:
            raise ValueError("equilibrium_id must be non-empty canonical text.")
        for name, value in (
            ("rho_faces", rho),
            ("contours_rz_m", contours),
            ("enclosed_volume_m3", volume),
            ("surface_area_m2", area),
            ("major_radius_m", major),
            ("minor_radius_m", minor),
            ("safety_factor", safety),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "geometry_id",
            canonical_fingerprint(
                {
                    "kind": "flux-surface-geometry",
                    "rho_faces": array_tree_fingerprint(rho),
                    "contours": array_tree_fingerprint(contours),
                    "volume": array_tree_fingerprint(volume),
                    "area": array_tree_fingerprint(area),
                    "major_radius": array_tree_fingerprint(major),
                    "minor_radius": array_tree_fingerprint(minor),
                    "safety_factor": array_tree_fingerprint(safety),
                    "equilibrium": equilibrium,
                    "topology": "nested-star-shaped",
                }
            ),
        )

    @property
    def cell_volume_m3(self) -> np.ndarray:
        result = np.diff(self.enclosed_volume_m3)
        result.setflags(write=False)
        return result

    def prepare(self) -> PreparedFluxSurfaceGeometry:
        metric_line = MetricLinePlan(
            self.rho_faces,
            self.cell_volume_m3,
            self.surface_area_m2,
            self.geometry_id,
        ).prepare()
        contour_gap = np.linalg.norm(
            self.contours_rz_m - np.roll(self.contours_rz_m, 1, axis=1), axis=-1
        )
        evidence = FluxSurfaceEvidence(
            jnp.asarray(np.min(self.cell_volume_m3)),
            jnp.asarray(np.max(contour_gap)),
            jnp.asarray(True),
            jnp.asarray(True),
            jnp.asarray(True),
        )
        rho = jnp.asarray(self.rho_faces)
        return PreparedFluxSurfaceGeometry(
            rho,
            0.5 * (rho[:-1] + rho[1:]),
            jnp.asarray(self.contours_rz_m),
            jnp.asarray(self.enclosed_volume_m3),
            jnp.asarray(self.cell_volume_m3),
            jnp.asarray(self.surface_area_m2),
            jnp.asarray(self.major_radius_m),
            jnp.asarray(self.minor_radius_m),
            jnp.asarray(self.safety_factor),
            metric_line,
            evidence,
            self.equilibrium_id,
            self.geometry_id,
            "fixed-prepared-topology",
        )


@dataclass(frozen=True, slots=True)
class FluxSurfacePlan:
    """Fixed-capacity ray preparation for nested star-shaped equilibria."""

    rho_faces: np.ndarray
    poloidal_count: int = 128
    radial_search_count: int = 512
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        rho = np.array(self.rho_faces, dtype=np.float64, copy=True)
        if (
            rho.ndim != 1
            or rho.size < 2
            or rho[0] != 0.0
            or np.any(np.diff(rho) <= 0.0)
            or rho[-1] >= 1.0
        ):
            raise ValueError("rho_faces must start at zero, increase, and end below one.")
        counts = (self.poloidal_count, self.radial_search_count)
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in counts):
            raise TypeError("Flux-surface preparation counts must be integers.")
        poloidal, radial = (int(value) for value in counts)
        if poloidal < 16 or radial < 32:
            raise ValueError(
                "Flux-surface preparation requires at least 16 angles and 32 rays."
            )
        rho.setflags(write=False)
        object.__setattr__(self, "rho_faces", rho)
        object.__setattr__(self, "poloidal_count", poloidal)
        object.__setattr__(self, "radial_search_count", radial)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "flux-surface-plan",
                    "rho_faces": array_tree_fingerprint(rho),
                    "poloidal_count": poloidal,
                    "radial_search_count": radial,
                    "topology": "nested-star-shaped",
                }
            ),
        )

    def prepare(self, equilibrium: AxisymmetricEquilibrium, /) -> FluxSurfaceGeometry:
        if not isinstance(equilibrium, AxisymmetricEquilibrium):
            raise TypeError("equilibrium must be AxisymmetricEquilibrium.")
        axis = equilibrium.magnetic_axis_rz_m
        normalized = equilibrium.normalized_flux
        axis_value = _bilinear(
            equilibrium.r_m, equilibrium.z_m, normalized, axis[0], axis[1]
        )
        if abs(axis_value) > 5.0e-3:
            raise ValueError(
                "Interpolated magnetic-axis flux is inconsistent with equilibrium metadata."
            )
        angles = 2.0 * math.pi * np.arange(self.poloidal_count) / self.poloidal_count
        contours = np.empty(
            (self.rho_faces.size, self.poloidal_count, 2), dtype=np.float64
        )
        contours[0, :, :] = axis
        for face_index, rho in enumerate(self.rho_faces[1:], start=1):
            target = rho * rho
            for angle_index, angle in enumerate(angles):
                direction = np.asarray([math.cos(angle), math.sin(angle)])
                contours[face_index, angle_index] = _ray_crossing(
                    equilibrium.r_m,
                    equilibrium.z_m,
                    normalized,
                    axis,
                    direction,
                    target,
                    self.radial_search_count,
                )
        volume = np.zeros((self.rho_faces.size,), dtype=np.float64)
        surface_area = np.zeros_like(volume)
        major_radius = np.full_like(volume, axis[0])
        minor_radius = np.zeros_like(volume)
        for index in range(1, self.rho_faces.size):
            polygon = contours[index]
            area, centroid_r = _polygon_area_centroid_r(polygon)
            volume[index] = 2.0 * math.pi * centroid_r * area
            differences = np.roll(polygon, -1, axis=0) - polygon
            midpoint_r = 0.5 * (polygon[:, 0] + np.roll(polygon[:, 0], -1))
            surface_area[index] = (
                2.0 * math.pi * np.sum(midpoint_r * np.linalg.norm(differences, axis=1))
            )
            major_radius[index] = centroid_r
            minor_radius[index] = math.sqrt(area / math.pi)
        if np.any(np.diff(volume) <= 0.0):
            raise ValueError("Prepared flux surfaces are not strictly nested by volume.")
        profile_coordinate = np.linspace(0.0, 1.0, equilibrium.safety_factor.size)
        safety = np.interp(
            self.rho_faces**2, profile_coordinate, equilibrium.safety_factor
        )
        geometry = FluxSurfaceGeometry(
            self.rho_faces,
            contours,
            volume,
            surface_area,
            major_radius,
            minor_radius,
            safety,
            equilibrium.equilibrium_id,
        )
        return geometry


def _bilinear(r, z, values, query_r: float, query_z: float) -> float:
    if query_r < r[0] or query_r > r[-1] or query_z < z[0] or query_z > z[-1]:
        raise ValueError("Flux interpolation query lies outside the R-Z grid.")
    i = int(np.clip(np.searchsorted(r, query_r, side="right") - 1, 0, r.size - 2))
    j = int(np.clip(np.searchsorted(z, query_z, side="right") - 1, 0, z.size - 2))
    wr = (query_r - r[i]) / (r[i + 1] - r[i])
    wz = (query_z - z[j]) / (z[j + 1] - z[j])
    return float(
        (1.0 - wr) * (1.0 - wz) * values[j, i]
        + wr * (1.0 - wz) * values[j, i + 1]
        + (1.0 - wr) * wz * values[j + 1, i]
        + wr * wz * values[j + 1, i + 1]
    )


def _ray_limit(r, z, origin, direction) -> float:
    candidates = []
    if direction[0] > 0.0:
        candidates.append((r[-1] - origin[0]) / direction[0])
    elif direction[0] < 0.0:
        candidates.append((r[0] - origin[0]) / direction[0])
    if direction[1] > 0.0:
        candidates.append((z[-1] - origin[1]) / direction[1])
    elif direction[1] < 0.0:
        candidates.append((z[0] - origin[1]) / direction[1])
    positive = tuple(value for value in candidates if value > 0.0)
    if not positive:
        raise ValueError("Flux-surface ray has no positive grid-boundary intersection.")
    return min(positive)


def _ray_crossing(r, z, values, origin, direction, target, search_count):
    maximum = _ray_limit(r, z, origin, direction)
    distances = np.linspace(0.0, maximum * (1.0 - 1.0e-12), search_count)
    samples = np.asarray(
        [
            _bilinear(r, z, values, *(origin + distance * direction))
            for distance in distances
        ]
    )
    crossing = np.flatnonzero((samples[:-1] < target) & (samples[1:] >= target))
    if crossing.size == 0:
        raise ValueError(
            "Requested flux surface is not star-shaped and closed on every ray."
        )
    lower = distances[int(crossing[0])]
    upper = distances[int(crossing[0]) + 1]
    for _ in range(56):
        middle = 0.5 * (lower + upper)
        value = _bilinear(r, z, values, *(origin + middle * direction))
        if value < target:
            lower = middle
        else:
            upper = middle
    return origin + 0.5 * (lower + upper) * direction


def _polygon_area_centroid_r(points):
    next_points = np.roll(points, -1, axis=0)
    cross = points[:, 0] * next_points[:, 1] - next_points[:, 0] * points[:, 1]
    signed_area = 0.5 * np.sum(cross)
    if not np.isfinite(signed_area) or abs(signed_area) <= np.finfo(float).eps:
        raise ValueError("Flux-surface polygon has zero or nonfinite area.")
    centroid_r = np.sum((points[:, 0] + next_points[:, 0]) * cross) / (6.0 * signed_area)
    return abs(float(signed_area)), float(centroid_r)


__all__ = [
    "FluxSurfaceEvidence",
    "FluxSurfaceGeometry",
    "FluxSurfacePlan",
    "PreparedFluxSurfaceGeometry",
]
