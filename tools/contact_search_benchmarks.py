#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measured deterministic dense and sweep-and-prune contact search."""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _scene(segment_count):
    x = jnp.linspace(0.0, 1.0, segment_count + 1)
    positions = jnp.stack((x, 0.01 * jnp.sin(20.0 * x)), axis=-1)
    edges = jnp.stack(
        (jnp.arange(segment_count), jnp.arange(1, segment_count + 1)), axis=-1
    )
    source = phx.linalg.ArraySpace(positions.shape, dtype=np.float64)
    plan = phx.discretization.CollisionSurfacePlan(
        jnp.arange(segment_count + 1, dtype=jnp.int64),
        ambient_dimension=2,
        edges=edges,
    )
    surface = phx.discretization.PreparedCollisionSurface(
        plan,
        positions,
        phx.discretization.selection_collision_operator(
            source, jnp.arange(segment_count + 1, dtype=jnp.int32)
        ),
    )
    return source, phx.discretization.PreparedCollisionScene((surface,))


def _time(plan, scene, positions, repeats):
    started = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = plan.build(scene, positions)
    elapsed = (time.perf_counter() - started) / repeats
    return result, elapsed


def main():
    source, scene = _scene(256)
    positions = scene.positions(source.zeros())
    capacities = dict(
        edge_vertex_capacity=4096,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.005,
    )
    dense, dense_seconds = _time(
        phx.discretization.DenseContactSearchPlan(**capacities),
        scene,
        positions,
        3,
    )
    sweep, sweep_seconds = _time(
        phx.discretization.SweepAndPruneContactSearchPlan(**capacities),
        scene,
        positions,
        20,
    )
    print(
        json.dumps(
            {
                "benchmark": "contact-search",
                "device": str(jax.devices()[0]),
                "dtype": str(positions.dtype),
                "vertices": scene.vertex_count,
                "edges": scene.edge_count,
                "candidate_count": int(sweep.candidate_count),
                "dense_seconds": dense_seconds,
                "sweep_seconds": sweep_seconds,
                "sets_equal": sorted(
                    np.asarray(
                        dense.edge_vertex.route_keys[dense.edge_vertex.valid]
                    ).tolist()
                )
                == sorted(
                    np.asarray(
                        sweep.edge_vertex.route_keys[sweep.edge_vertex.valid]
                    ).tolist()
                ),
                "successful": bool(dense.successful & sweep.successful),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
