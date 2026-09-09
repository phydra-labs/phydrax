#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Deterministic performance and invariant benchmarks for neurofluid substrates."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def image_sampling(size: int, query_count: int, repeats: int) -> dict[str, float | int]:
    contract = phx.SpatialCoordinateContract(
        phx.units.MILLIMETER,
        coordinate_system="cartesian-lps",
        reference_frame="benchmark",
    )
    affine = phx.imaging.ImageIndexAffine(
        np.eye(4), "voxels", contract, phx.imaging.ImageAxisConvention.LPS
    )
    axis = np.linspace(0.1, size - 1.1, query_count)
    points = np.column_stack(
        (axis, np.mod(2.0 * axis, size - 1), np.mod(3.0 * axis, size - 1))
    )
    start = perf_counter()
    prepared = phx.spatial_sampling.VoxelObservationPlan(
        (size, size, size), affine, points, require_complete_coverage=True
    ).prepare()
    preparation = perf_counter() - start
    values = jnp.asarray(
        np.fromfunction(lambda i, j, k: i + 2.0 * j + 3.0 * k, (size, size, size))
    )
    execute = jax.jit(lambda operator, data: operator.apply(data).values)
    start = perf_counter()
    first_values = execute(prepared, values)
    jax.block_until_ready(first_values)
    compilation = perf_counter() - start
    start = perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(execute(prepared, values))
    elapsed = (perf_counter() - start) / repeats
    first = prepared.apply(values)
    expected = points[:, 0] + 2.0 * points[:, 1] + 3.0 * points[:, 2]
    error = float(np.max(np.abs(np.asarray(first_values) - expected)))
    if error > 1.0e-10 or not bool(first.evidence.successful):
        raise RuntimeError("Image sampling benchmark violated affine-field exactness.")
    route_bytes = sum(value.nbytes for value in jax.tree.leaves(prepared.stencil))
    return {
        "size": size,
        "query_count": query_count,
        "preparation_seconds": preparation,
        "compile_seconds": compilation,
        "execution_seconds": elapsed,
        "route_bytes": route_bytes,
        "maximum_error": error,
    }


def conservative_transfer(count: int, repeats: int) -> dict[str, float | int]:
    indices = np.arange(count, dtype=np.int32)
    measures = np.ones((count,))
    transfer = phx.imaging.ConservativeVoxelCellTransfer(
        indices,
        indices,
        measures,
        measures,
        measures,
        source_id="benchmark-voxels",
        target_id="benchmark-cells",
    )
    values = jnp.linspace(0.0, 1.0, count)
    apply = jax.jit(lambda operator, data: operator.apply(data).values)
    start = perf_counter()
    first = apply(transfer, values)
    jax.block_until_ready(first)
    compilation = perf_counter() - start
    start = perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(apply(transfer, values))
    elapsed = (perf_counter() - start) / repeats
    error = float(jnp.max(jnp.abs(first - values)))
    evidence = transfer.apply(values).evidence
    if error > 1.0e-12 or not bool(evidence.successful):
        raise RuntimeError("Conservative transfer benchmark violated identity and mass.")
    storage = sum(value.nbytes for value in jax.tree.leaves(transfer))
    return {
        "entity_count": count,
        "compile_seconds": compilation,
        "execution_seconds": elapsed,
        "storage_bytes": storage,
        "maximum_error": error,
    }


def hdiv_tabulation(repeats: int) -> dict[str, float | int]:
    element = phx.discretization.tetrahedral_bdm_element(2)
    points = jnp.asarray(((0.1, 0.2, 0.3), (0.25, 0.25, 0.25)))
    if element.tabulator is None:
        raise RuntimeError("BDM benchmark requires its native reference tabulator.")
    tabulate = jax.jit(element.tabulator)
    values, gradients = tabulate(points)
    jax.block_until_ready(values)
    start = perf_counter()
    for _ in range(repeats):
        values, gradients = tabulate(points)
        jax.block_until_ready(gradients)
    elapsed = (perf_counter() - start) / repeats
    divergence = jnp.trace(gradients, axis1=-2, axis2=-1)
    if not bool(jnp.all(jnp.isfinite(divergence))):
        raise RuntimeError("BDM benchmark produced non-finite divergence.")
    return {
        "dofs": element.local_dof_count,
        "point_count": len(points),
        "execution_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.size < 2 or args.queries < 1 or args.repeats < 1:
        parser.error("size, queries, and repeats must be positive; size must exceed one")
    print(
        json.dumps(
            {
                "image_sampling": image_sampling(args.size, args.queries, args.repeats),
                "conservative_transfer": conservative_transfer(
                    args.queries, args.repeats
                ),
                "hdiv_tabulation": hdiv_tabulation(args.repeats),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
