#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Portable timing evidence for dynamic triangle and point rendering kernels."""

from __future__ import annotations

import argparse
import json

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import capture_environment, measure_repeated


def _triangles(count: int):
    columns = int(np.ceil(np.sqrt(count)))
    vertices = []
    triangles = []
    for index in range(count):
        row, column = divmod(index, columns)
        center_x = (column - columns / 2) * 0.1
        center_y = (row - columns / 2) * 0.1
        base = len(vertices)
        vertices.extend(
            (
                (center_x - 0.04, center_y - 0.04, 2.0),
                (center_x + 0.04, center_y - 0.04, 2.0),
                (center_x, center_y + 0.04, 2.0),
            )
        )
        triangles.append((base, base + 1, base + 2))
    return np.asarray(vertices), np.asarray(triangles, dtype=np.int32)


def benchmark(*, smoke: bool) -> dict[str, object]:
    count = 64 if smoke else 1_024
    repeats = 2 if smoke else 10
    vertices, triangles = _triangles(count)
    query = phx.geometry.prepare_triangle_ray_query(
        phx.geometry.TriangleRayQueryPlan(
            vertices,
            triangles,
            leaf_size=8,
            traversal_stack_capacity=64,
        )
    )
    origins = jnp.zeros((count, 3))
    centers = jnp.asarray(vertices[triangles].mean(axis=1))
    directions = centers / jnp.sqrt(jnp.sum(centers * centers, axis=1, keepdims=True))

    def dynamic_query(current_vertices):
        geometry = phx.geometry.refit_triangle_ray_geometry(
            query, current_vertices, geometry_id="benchmark-geometry"
        )
        return phx.geometry.intersect_triangle_rays(
            query, origins, directions, geometry=geometry
        )

    compiled_query = eqx.filter_jit(dynamic_query)
    hit, ray_distribution = measure_repeated(
        lambda: compiled_query(jnp.asarray(vertices)),
        warmup=1,
        repeats=repeats,
    )
    support = phx.imaging.ImagePlaneSupport((64, 64))
    rasterizer = phx.rendering.GaussianRasterizer(6, cutoff=3.0)
    positions = jnp.asarray(((16.25, 16.5), (32.5, 31.75), (48.0, 48.0)))
    compiled_raster = eqx.filter_jit(rasterizer.render)
    raster, raster_distribution = measure_repeated(
        lambda: compiled_raster(
            support,
            positions,
            jnp.asarray((1.0, 2.0, 3.0)),
            jnp.ones((3,)),
            jnp.ones((3,), dtype=bool),
        ),
        warmup=1,
        repeats=repeats,
    )
    if not bool(jnp.all(hit.successful)) or not bool(raster.successful):
        raise RuntimeError("Rendering benchmark evidence was unsuccessful.")
    return {
        "environment": capture_environment().to_dict(),
        "configuration": {"triangle_count": count, "repeats": repeats},
        "dynamic_triangle_query": ray_distribution.to_milliseconds_dict(),
        "gaussian_raster": raster_distribution.to_milliseconds_dict(),
        "successful": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(benchmark(smoke=arguments.smoke), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
