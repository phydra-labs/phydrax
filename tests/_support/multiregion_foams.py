#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic dry-foam multiregion seeds for topology-event tests."""

from __future__ import annotations

import numpy as np

from phydrax.geometry.multiregion_surface import (
    MultiRegionSurfaceCapacityPlan,
    MultiRegionSurfaceSeed,
)


def dry_foam_capacity_plan(
    seed: MultiRegionSurfaceSeed, resource: str, /
) -> MultiRegionSurfaceCapacityPlan:
    """Doubled entity capacities with dry-foam valence and slot headroom."""
    counts = seed.counts()
    return MultiRegionSurfaceCapacityPlan(
        vertex_capacity=2 * counts.vertex,
        edge_capacity=2 * counts.edge,
        face_capacity=2 * counts.face,
        region_capacity=counts.region + 2,
        region_pair_capacity=2 * counts.region_pair + 2,
        maximum_edge_valence=3,
        maximum_vertex_region_pairs=9,
        resource_id=resource,
        event_capacity=16,
    )


def t1_cluster(
    film: float = 0.05, cap: float = 0.25, height: float = 0.5, turn: float = 0.0
) -> MultiRegionSurfaceSeed:
    """Cells D (upper column), E (lower column) and three wedges around a tiny D|E film.

    The column is a frustum whose mid-plane triangle (the vanishing film,
    circumradius ``film``) is bounded by three Plateau borders; slight
    deterministic offsets keep every vertex in general position.
    """
    angles = np.radians(90.0 + turn + 120.0 * np.arange(3))
    ring = np.stack((np.cos(angles), np.sin(angles), np.zeros(3)), axis=1)
    points: list[np.ndarray] = []

    def vertex(point: np.ndarray, key: int, level: int, /) -> int:
        offset = 1.3e-3 * np.asarray(
            (np.sin(3.1 * key + 7.0 * level), np.cos(2.3 * key + 5.0 * level), 0.0)
        )
        points.append(point + offset)
        return len(points) - 1

    inner = {
        (k, s): vertex((film if s == 0 else cap) * ring[k] + (0.0, 0.0, s * height), k, s)
        for k in range(3)
        for s in (-1, 0, 1)
    }
    outer = {
        (k, s): vertex(ring[k] + (0.0, 0.0, s * height), k + 5, s)
        for k in range(3)
        for s in (-1, 0, 1)
    }
    upper, lower, ambient = 0, 1, 5
    wedge = (2, 3, 4)
    centers = {
        upper: np.asarray((0.0, 0.0, 0.5 * height)),
        lower: np.asarray((0.0, 0.0, -0.5 * height)),
    }
    for k in range(3):
        centers[wedge[k]] = 0.3 * (ring[k] + ring[(k + 1) % 3])
    polygons = [
        ([inner[(k, 0)] for k in range(3)], upper, lower),
        ([inner[(k, 1)] for k in range(3)], upper, ambient),
        ([inner[(k, -1)] for k in range(3)], lower, ambient),
    ]
    for k in range(3):
        n = (k + 1) % 3
        side, previous = wedge[k], wedge[(k - 1) % 3]
        polygons += [
            ([inner[(k, 0)], inner[(n, 0)], inner[(n, 1)], inner[(k, 1)]], upper, side),
            ([inner[(k, -1)], inner[(n, -1)], inner[(n, 0)], inner[(k, 0)]], lower, side),
            ([inner[(k, 1)], outer[(k, 1)], outer[(n, 1)], inner[(n, 1)]], side, ambient),
            (
                [inner[(k, -1)], outer[(k, -1)], outer[(n, -1)], inner[(n, -1)]],
                side,
                ambient,
            ),
            (
                [outer[(k, -1)], outer[(n, -1)], outer[(n, 0)], outer[(k, 0)]],
                side,
                ambient,
            ),
            ([outer[(k, 0)], outer[(n, 0)], outer[(n, 1)], outer[(k, 1)]], side, ambient),
            ([inner[(k, -1)], outer[(k, -1)], inner[(k, 0)]], previous, side),
            ([inner[(k, 0)], outer[(k, -1)], outer[(k, 0)]], previous, side),
            ([inner[(k, 0)], outer[(k, 0)], outer[(k, 1)]], previous, side),
            ([inner[(k, 0)], outer[(k, 1)], inner[(k, 1)]], previous, side),
        ]
    coordinates = np.asarray(points)
    faces, labels = [], []
    for loop, left, right in polygons:
        for index in range(1, len(loop) - 1):
            triangle = [loop[0], loop[index], loop[index + 1]]
            corners = coordinates[triangle]
            normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
            if np.dot(normal, np.mean(corners, axis=0) - centers[left]) < 0.0:
                triangle = [triangle[0], triangle[2], triangle[1]]
            faces.append(triangle)
            labels.append((left, right))
    return MultiRegionSurfaceSeed(
        coordinates,
        np.asarray(faces),
        np.asarray(labels),
        ("D", "E", "A", "B", "C", "ambient"),
        ("finite",) * 5 + ("boundary",),
        source="t1-cluster",
    )
