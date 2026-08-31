#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _triangular_grid(width):
    axis = np.linspace(-1.0, 1.0, width)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    vertices = np.stack((x.reshape((-1,)), y.reshape((-1,))), axis=1)
    faces = []
    for row in range(width - 1):
        for column in range(width - 1):
            lower_left = row * width + column
            lower_right = lower_left + 1
            upper_left = lower_left + width
            upper_right = upper_left + 1
            faces.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    return jnp.asarray(vertices), jnp.asarray(faces, dtype=jnp.int32)


def _timed(function):
    started = time.perf_counter()
    value = function()
    return value, time.perf_counter() - started


def topology_case(width, repeats):
    vertices, faces = _triangular_grid(width)
    mesh = phx.discretization.CellMesh.from_triangles(vertices, faces)
    complex = phx.topology.CellSubcomplex.full(mesh.topology)
    support = phx.topology.cell_vertex_support(
        mesh.topology,
        (
            np.arange(vertices.shape[0], dtype=np.int32)[:, None],
            np.asarray(mesh.connectivity.edges, dtype=np.int32),
            np.asarray(faces, dtype=np.int32),
        ),
    )
    field = jnp.linalg.norm(vertices, axis=1)
    filtration = phx.topology.lower_star_filtration(
        complex,
        support,
        field,
        source_id=f"radial-grid-{width}",
    )
    homology, homology_seconds = _timed(
        lambda: phx.topology.compute_homology(
            complex,
            coefficients=phx.topology.PrimeField(2),
        )
    )
    rational, rational_seconds = _timed(
        lambda: phx.topology.compute_betti_dimensions(
            complex,
            coefficients=phx.topology.RationalField(),
        )
    )
    persistence, persistence_seconds = _timed(
        lambda: phx.topology.compute_persistence(
            filtration,
            coefficients=phx.topology.PrimeField(2),
        )
    )
    frozen = phx.topology.freeze_persistence_pairing(persistence, filtration)
    evaluate = eqx.filter_jit(frozen.evaluate)
    evaluate(filtration.values).birth_values.block_until_ready()
    started = time.perf_counter()
    for _ in range(repeats):
        evaluated = evaluate(filtration.values)
        evaluated.birth_values.block_until_ready()
    frozen_seconds = (time.perf_counter() - started) / repeats
    return {
        "vertices": int(vertices.shape[0]),
        "edges": int(mesh.topology.entity_sets[1].count),
        "faces": int(faces.shape[0]),
        "boundary_nonzeros": int(
            sum(incidence.relation.capacity for incidence in mesh.topology.incidences)
        ),
        "homology": list(homology.dimensions),
        "rational_betti": list(rational.dimensions),
        "intervals": int(persistence.diagram().interval_count),
        "homology_seconds": homology_seconds,
        "rational_seconds": rational_seconds,
        "persistence_seconds": persistence_seconds,
        "frozen_steady_seconds": frozen_seconds,
        "homology_operations": dict(homology.evidence.counts)["operations"],
        "persistence_operations": dict(persistence.evidence.counts)["operations"],
        "persistence_peak_entries": dict(persistence.evidence.counts)[
            "peak_reduction_entries"
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-width", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.mesh_width < 2 or arguments.repeats < 1:
        raise ValueError("Benchmark sizes and repeats are below their valid minimum.")
    print(
        json.dumps(
            topology_case(arguments.mesh_width, arguments.repeats),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
