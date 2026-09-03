from __future__ import annotations

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _block(value) -> None:
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _measure(function, *arguments):
    started = time.perf_counter()
    value = function(*arguments)
    _block(value)
    return value, time.perf_counter() - started


def _case(depth: int):
    resolution = 1 << depth
    address = phx.discretization.MortonAddressPlan(
        (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), depth
    )
    integer = np.stack(
        np.meshgrid(
            np.arange(resolution),
            np.arange(resolution),
            np.arange(resolution),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((-1, 3))
    z_coordinate = (integer[:, 2] + 0.5) * 2.0 / resolution - 1.0
    active_coordinates = integer[np.abs(z_coordinate) <= 0.35]
    brick_size = 4
    brick_count = np.unique(active_coordinates // brick_size, axis=0).shape[0]
    grid = phx.discretization.SparseVoxelGridPlan(
        address,
        brick_size=brick_size,
        brick_capacity=brick_count,
    ).prepare(active_coordinates)
    axis = jnp.linspace(-0.75, 0.75, 4)
    first, second = jnp.meshgrid(axis, axis, indexing="ij")
    positions = jnp.stack(
        (first.reshape((-1,)), second.reshape((-1,)), jnp.zeros((16,))),
        axis=-1,
    )
    normals = jnp.tile(jnp.asarray((0.0, 0.0, 1.0)), (16, 1))
    radius = 0.38
    axes = jnp.tile(
        jnp.asarray(((radius, 0.0), (0.0, radius), (0.0, 0.0)))[None, ...],
        (16, 1, 1),
    )
    weights = jnp.full((16,), 0.25)
    surfels = phx.discretization.SurfelSetPlan(
        jnp.arange(16, dtype=jnp.int64), positions, weights
    ).prepare()
    certificate = phx.discretization.SurfelGeometryCertificate(
        source_geometry_id="benchmark-plane",
        source_kind="analytic",
        orientation_scope=phx.discretization.SurfelOrientationScope.GLOBAL,
    )
    geometry = phx.discretization.SurfelGeometryPlan(surfels).materialize(
        positions, normals, axes, certificate=certificate
    )
    voxel_width = 2.0 / resolution
    maximum_axis_span = int(np.ceil((2.0 * radius) / voxel_width)) + 3
    maximum_candidates = maximum_axis_span**3
    route_capacity = 16 * maximum_candidates
    plan = phx.discretization.SurfelVoxelProjectionPlan(
        grid,
        geometry,
        maximum_voxels_per_surfel=maximum_candidates,
        route_capacity=route_capacity,
        normal_distance_support=0.3,
        route_padding=voxel_width,
    )
    prepare = eqx.filter_jit(plan.prepare)
    prepared, prepare_first = _measure(prepare, geometry)
    _, prepare_steady = _measure(prepare, geometry)
    project = eqx.filter_jit(prepared.project)
    result, project_first = _measure(project, geometry)
    _, project_steady = _measure(project, geometry)
    centers = grid.voxel_centers()
    supported = result.supported
    error = jnp.max(
        jnp.where(
            supported,
            jnp.abs(result.implicit_value - centers[..., 2]),
            0.0,
        ),
        initial=0.0,
    )
    return {
        "depth": depth,
        "resolution": resolution,
        "surfels": 16,
        "active_voxels": int(grid.evidence.active_voxels),
        "required_routes": int(prepared.evidence.required_routes),
        "route_capacity": route_capacity,
        "maximum_candidates_per_surfel": int(
            prepared.evidence.maximum_candidates_per_surfel
        ),
        "candidate_capacity": maximum_candidates,
        "candidate_overflow": bool(prepared.evidence.candidate_overflow),
        "route_overflow": bool(prepared.evidence.route_overflow),
        "active_projection_routes": int(result.evidence.active_routes),
        "supported_voxels": int(result.evidence.supported_voxels),
        "stale_surfels": int(result.evidence.stale_surfels),
        "prepare_first_seconds": prepare_first,
        "prepare_steady_seconds": prepare_steady,
        "project_first_seconds": project_first,
        "project_steady_seconds": project_steady,
        "maximum_plane_error": float(error),
        "successful": bool(result.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark surfel-to-voxel projection.")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    depths = (4,) if arguments.smoke else (4, 5)
    cases = [_case(depth) for depth in depths]
    report = {
        "kind": "surfel-voxel-benchmark",
        "device": str(jax.devices()[0]),
        "cases": cases,
        "passed": all(
            case["successful"] and case["maximum_plane_error"] < 1.0e-10 for case in cases
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
