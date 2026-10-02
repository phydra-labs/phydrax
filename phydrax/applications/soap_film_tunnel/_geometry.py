#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Planar soap-film tunnel channel and its constrained Delaunay triangulation.

The film lies in the ``z = 0`` plane on ``[0, L] x [0, W]``. The tunnel axis
``+x`` points downstream (downward in a vertical gravity-driven tunnel): the
film enters at ``x = 0`` and leaves at ``x = L`` between the wires at
``y = 0`` and ``y = W``. An optional circular obstacle is a hole in the film
whose rim is a polygon of ``rim_segments`` chords.

The channel is triangulated by the native constrained Delaunay owner
(``phydrax.geometry.ConstrainedDelaunayTriangulation``, Ruppert/Chew
refinement; requires the optional ``phydrax-meshcore`` library). Refinement
leaves no input segment encroached, so the angles opposite boundary edges are
acute, and the free edges are Delaunay: the cotangent conductances of the
film route are nonnegative, which ``PreparedFilmSurface`` audits. Boundary
parts are read from the preserved input-segment labels, never from
coordinate tolerances.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float, positive_integer
from ...geometry import ConstrainedDelaunayTriangulation, TriangulationEvidence
from ...geometry.simplicial import TriangleMesh
from ...typing import checked


_INLET, _OUTLET, _WIRE, _RIM = 0, 1, 2, 3


class SoapFilmTunnelMesh(StrictModule, NonTrainableState):
    """Triangulated channel with its boundary parts and triangulation evidence.

    Vertex id arrays name the vertices on each boundary part; corner vertices
    belong to both adjacent parts.
    """

    mesh: TriangleMesh
    inlet_vertices: Array
    outlet_vertices: Array
    wire_vertices: Array
    rim_vertices: Array
    triangulation: TriangulationEvidence
    mesh_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: TriangleMesh,
        inlet_vertices: np.ndarray,
        outlet_vertices: np.ndarray,
        wire_vertices: np.ndarray,
        rim_vertices: np.ndarray,
        triangulation: TriangulationEvidence,
        /,
        *,
        geometry_id: str,
    ) -> None:
        self.mesh = mesh
        self.inlet_vertices = jnp.asarray(inlet_vertices, dtype=jnp.int32)
        self.outlet_vertices = jnp.asarray(outlet_vertices, dtype=jnp.int32)
        self.wire_vertices = jnp.asarray(wire_vertices, dtype=jnp.int32)
        self.rim_vertices = jnp.asarray(rim_vertices, dtype=jnp.int32)
        self.triangulation = triangulation
        self.mesh_id = canonical_fingerprint(
            {
                "kind": "soap-film-tunnel-mesh",
                "geometry_id": geometry_id,
                "triangulation_id": triangulation.evidence_id,
            }
        )


class SoapFilmTunnelGeometry(StrictModule):
    """Channel size, optional circular obstacle and triangulation controls.

    ``mesh_size_m`` is the boundary spacing of the inlet, outlet and wires and
    sets the largest triangle area ``sqrt(3)/4 mesh_size^2``; the rim spacing
    is ``pi d / rim_segments``. Refinement grades the mesh between the two.
    ``minimum_angle_degrees`` is the Ruppert/Chew quality target and
    ``maximum_steiner_points`` bounds the inserted points.
    """

    length_m: float = eqx.field(static=True)
    width_m: float = eqx.field(static=True)
    mesh_size_m: float = eqx.field(static=True)
    obstacle_diameter_m: float | None = eqx.field(static=True)
    obstacle_center_m: tuple[float, float] | None = eqx.field(static=True)
    rim_segments: int = eqx.field(static=True)
    minimum_angle_degrees: float = eqx.field(static=True)
    maximum_steiner_points: int = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        length_m: float,
        width_m: float,
        /,
        *,
        mesh_size_m: float,
        obstacle_diameter_m: float | None = None,
        obstacle_center_m: tuple[float, float] | None = None,
        rim_segments: int = 48,
        minimum_angle_degrees: float = 28.0,
        maximum_steiner_points: int = 1_000_000,
    ) -> None:
        length = positive_finite_float(length_m, "length_m")
        width = positive_finite_float(width_m, "width_m")
        size = positive_finite_float(mesh_size_m, "mesh_size_m")
        if 2.0 * size > min(length, width):
            raise ValueError("mesh_size_m must resolve every channel side twice.")
        segments = positive_integer(rim_segments, "rim_segments")
        if segments < 8:
            raise ValueError("rim_segments must be at least 8.")
        angle = positive_finite_float(minimum_angle_degrees, "minimum_angle_degrees")
        if angle > 33.0:
            raise ValueError("minimum_angle_degrees above 33 does not terminate.")
        steiner = positive_integer(maximum_steiner_points, "maximum_steiner_points")
        if (obstacle_diameter_m is None) != (obstacle_center_m is None):
            raise ValueError("Obstacle diameter and center must be given together.")
        diameter = (
            None
            if obstacle_diameter_m is None
            else positive_finite_float(obstacle_diameter_m, "obstacle_diameter_m")
        )
        center = None
        if obstacle_center_m is not None and diameter is not None:
            center = _center(obstacle_center_m)
            clearance = (
                min(center[0], length - center[0], center[1], width - center[1])
                - 0.5 * diameter
            )
            if clearance < size:
                raise ValueError(
                    "The obstacle must clear every channel side by one mesh size."
                )
        self.length_m = length
        self.width_m = width
        self.mesh_size_m = size
        self.obstacle_diameter_m = diameter
        self.obstacle_center_m = center
        self.rim_segments = segments
        self.minimum_angle_degrees = angle
        self.maximum_steiner_points = steiner
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "soap-film-tunnel-geometry",
                "length_m": length,
                "width_m": width,
                "mesh_size_m": size,
                "obstacle_diameter_m": diameter,
                "obstacle_center_m": center,
                "rim_segments": segments,
                "minimum_angle_degrees": angle,
                "maximum_steiner_points": steiner,
            }
        )

    @property
    def reference_length_m(self) -> float:
        """Obstacle diameter, or the channel width of an empty channel."""
        return (
            self.width_m if self.obstacle_diameter_m is None else self.obstacle_diameter_m
        )

    def triangulate(self) -> SoapFilmTunnelMesh:
        """Return the constrained Delaunay film mesh (requires phydrax-meshcore)."""
        points, segments, labels, holes = self._planar_graph()
        triangulation = ConstrainedDelaunayTriangulation(
            points,
            segments,
            holes=holes,
            min_angle=self.minimum_angle_degrees,
            max_area=0.25 * math.sqrt(3.0) * self.mesh_size_m**2,
            max_steiner=self.maximum_steiner_points,
        )
        triangles = np.asarray(triangulation.triangles, dtype=np.int32)
        if np.unique(triangles).size != triangulation.points.shape[0]:
            raise ValueError("The channel triangulation left isolated points.")
        edge_segment = np.asarray(triangulation.segment_ids)
        face, corner = np.nonzero(edge_segment >= 0)
        first = triangles[face, (corner + 1) % 3]
        second = triangles[face, (corner + 2) % 3]
        part = labels[edge_segment[face, corner]]

        def vertices(code: int, /) -> np.ndarray:
            selected = part == code
            return np.unique(np.concatenate((first[selected], second[selected])))

        mesh = TriangleMesh(
            np.column_stack(
                (triangulation.points, np.zeros((triangulation.points.shape[0],)))
            ),
            triangles,
            source_id=self.geometry_id,
        )
        return SoapFilmTunnelMesh(
            mesh,
            vertices(_INLET),
            vertices(_OUTLET),
            vertices(_WIRE),
            vertices(_RIM),
            triangulation.evidence,
            geometry_id=self.geometry_id,
        )

    def _planar_graph(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Return points, segments, segment part labels and hole seeds."""
        length, width = self.length_m, self.width_m
        along = math.ceil(length / self.mesh_size_m)
        across = math.ceil(width / self.mesh_size_m)
        corners = np.asarray(((0.0, 0.0), (length, 0.0), (length, width), (0.0, width)))
        # Counterclockwise: lower wire, outlet, upper wire, inlet.
        sides = ((along, _WIRE), (across, _OUTLET), (along, _WIRE), (across, _INLET))
        outer = []
        labels = []
        for index, (count, label) in enumerate(sides):
            start, end = corners[index], corners[(index + 1) % 4]
            fraction = np.arange(count, dtype=np.float64) / count
            outer.append(start[None] + fraction[:, None] * (end - start)[None])
            labels.append(np.full((count,), label, dtype=np.int32))
        points = np.concatenate(outer)
        loops = [points.shape[0]]
        holes = None
        if self.obstacle_diameter_m is not None and self.obstacle_center_m is not None:
            angle = 2.0 * np.pi * np.arange(self.rim_segments) / self.rim_segments
            radius = 0.5 * self.obstacle_diameter_m
            center = np.asarray(self.obstacle_center_m)
            # Clockwise hole loop.
            rim = center[None] + radius * np.column_stack((np.cos(angle), -np.sin(angle)))
            points = np.concatenate((points, rim))
            loops.append(self.rim_segments)
            labels.append(np.full((self.rim_segments,), _RIM, dtype=np.int32))
            holes = center[None]
        segments = []
        offset = 0
        for count in loops:
            ids = offset + np.arange(count, dtype=np.int32)
            segments.append(np.stack((ids, np.roll(ids, -1)), axis=1))
            offset += count
        return points, np.concatenate(segments), np.concatenate(labels), holes


def _center(value: tuple[float, float], /) -> tuple[float, float]:
    host = np.asarray(value, dtype=np.float64)
    if host.shape != (2,) or not np.all(np.isfinite(host)):
        raise ValueError("obstacle_center_m must be a finite (x, y) pair.")
    return (float(host[0]), float(host[1]))


__all__ = ["SoapFilmTunnelGeometry", "SoapFilmTunnelMesh"]
