#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import argparse
import json
from time import perf_counter

import jax
import numpy as np

import phydrax as phx


def benchmark(resolution: int, repeats: int) -> dict[str, object]:
    axis = np.linspace(0.0, 1.0, resolution + 1)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel()))
    lower = (
        np.arange(resolution)[:, None] * (resolution + 1) + np.arange(resolution)
    ).ravel()
    faces = np.concatenate(
        (
            np.column_stack((lower, lower + resolution + 1, lower + 1)),
            np.column_stack((lower + 1, lower + resolution + 1, lower + resolution + 2)),
        )
    )
    start = perf_counter()
    mesh = phx.discretization.CellMesh.from_triangles(points, faces)
    construction_seconds = perf_counter() - start
    start = perf_counter()
    result = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    certification_seconds = perf_counter() - start
    quality = jax.jit(
        lambda coordinates: phx.meshing.evaluate_cell_quality(mesh, coordinates)
    )
    start = perf_counter()
    jax.block_until_ready(quality(mesh.coordinates))
    quality_compile_seconds = perf_counter() - start
    start = perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(quality(mesh.coordinates))
    quality_seconds = (perf_counter() - start) / repeats
    start = perf_counter()
    transition, _ = phx.meshing.refine_triangle_mesh(
        mesh,
        np.asarray(mesh.blocks[0].global_ids)[:1],
        phx.SpatialCoordinateContract.si(),
    )
    refinement_seconds = perf_counter() - start
    stencil = transition.vertex_stencil
    assert stencil is not None
    transferred = stencil.apply(mesh.vertex_global_ids, mesh.coordinates[:, 0])
    error = float(
        np.max(
            np.abs(
                np.asarray(transferred)
                - np.asarray(transition.target.mesh.coordinates[:, 0])
            )
        )
    )
    if not result.audit.passed or error > 1e-12:
        raise RuntimeError(
            "Native meshing benchmark failed certification or linear transfer."
        )
    return {
        "resolution": resolution,
        "vertices": len(points),
        "cells": len(faces),
        "construction_seconds": construction_seconds,
        "certification_seconds": certification_seconds,
        "quality_compile_seconds": quality_compile_seconds,
        "quality_seconds": quality_seconds,
        "refinement_seconds": refinement_seconds,
        "linear_transfer_error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or any(value < 1 for value in args.resolution):
        parser.error("resolutions and repeats must be positive")
    print(
        json.dumps(
            {
                "native_meshing": [
                    benchmark(value, args.repeats) for value in args.resolution
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
