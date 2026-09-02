from __future__ import annotations

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _measure(function, *arguments):
    started = time.perf_counter()
    value = function(*arguments)
    jax.block_until_ready(value)
    return value, time.perf_counter() - started


def _case(depth: int, kind: str):
    resolution = 1 << depth
    coordinates = np.stack(
        np.meshgrid(
            np.arange(resolution),
            np.arange(resolution),
            np.arange(resolution),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((-1, 3))
    centered = (coordinates + 0.5) / resolution - 0.5
    if kind == "shell":
        radius = np.linalg.norm(centered, axis=1)
        coordinates = coordinates[np.abs(radius - 0.32) <= 2.0 / resolution]
    address = phx.discretization.MortonAddressPlan(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), depth
    )
    brick_size = 4
    brick_coordinates = np.unique(coordinates // brick_size, axis=0)
    plan = phx.discretization.SparseVoxelGridPlan(
        address,
        brick_size=brick_size,
        brick_capacity=max(int(brick_coordinates.shape[0]), 1),
    )
    started = time.perf_counter()
    grid = plan.prepare(coordinates)
    prepare_seconds = time.perf_counter() - started
    centers = grid.voxel_centers()
    values = jnp.sin(centers[..., 0]) + 2.0 * centers[..., 1] - centers[..., 2]
    field = phx.discretization.SparseVoxelField(grid, values)
    key = jax.random.key(depth)
    points = 0.1 + 0.8 * jax.random.uniform(key, (4096, 3))
    sample = eqx.filter_jit(field.sample_multilinear)
    first, first_seconds = _measure(sample, points)
    _, steady_seconds = _measure(sample, points)
    topology_bytes = sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(grid)
        if isinstance(leaf, jax.Array)
    )
    return {
        "kind": kind,
        "depth": depth,
        "resolution": resolution,
        "active_voxels": int(grid.evidence.active_voxels),
        "active_bricks": int(grid.evidence.active_bricks),
        "topology_bytes": topology_bytes,
        "prepare_seconds": prepare_seconds,
        "sample_first_seconds": first_seconds,
        "sample_steady_seconds": steady_seconds,
        "supported_queries": int(jnp.sum(first.supported)),
        "finite": bool(jnp.all(jnp.isfinite(first.values))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sparse voxel fields.")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    depths = (4,) if arguments.smoke else (4, 5, 6)
    cases = [_case(depth, kind) for depth in depths for kind in ("dense", "shell")]
    report = {
        "kind": "sparse-voxel-benchmark",
        "device": str(jax.devices()[0]),
        "cases": cases,
        "passed": all(case["finite"] for case in cases),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
