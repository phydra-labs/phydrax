#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _mesh():
    return phx.discretization.CellMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        (
            phx.discretization.CellBlock(
                "quad",
                "quadrilateral",
                jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
                global_ids=jnp.asarray((10,), dtype=jnp.int64),
            ),
        ),
    )


def benchmark(order: int) -> dict[str, float | int | bool]:
    capacity = 16
    topology, geometry = phx.discretization.fem.initial_finite_element_hp_topology(
        _mesh(), order, capacity
    )
    source = phx.discretization.fem.prepare_finite_element_hp_epoch(
        topology, geometry, "u", conformity="L2"
    )
    start = perf_counter()
    refined = phx.discretization.fem.refine_tensor_hp_cells(
        topology,
        geometry,
        jnp.asarray((10,), dtype=jnp.int64),
        target_degrees=jnp.asarray(((order + 1, order),), dtype=jnp.int32),
    )
    target = phx.discretization.fem.prepare_finite_element_hp_epoch(
        refined.topology,
        refined.geometry,
        "u",
        conformity="L2",
    )
    preparation_seconds = perf_counter() - start
    interfaces = phx.discretization.fem.finite_element_hp_interface_plan(
        refined.topology, refined.geometry
    )
    geometry_evidence = phx.discretization.fem.certify_finite_element_hp_geometry(
        refined.topology,
        refined.geometry,
        interfaces,
    )
    transfer = phx.discretization.fem.finite_element_hp_transfer_plan(
        source,
        target,
        refined.lineage,
        "u",
        "h-refinement",
    )
    values = jnp.zeros((capacity, transfer.primal.shape[2]))
    source_nodes = np.asarray(source.discretization.elements[0][0].reference_nodes)
    polynomial = source_nodes[:, 0] ** min(order, 2) + source_nodes[:, 1]
    values = values.at[0, : polynomial.size].set(polynomial)
    transferred = eqx.filter_jit(transfer.apply_primal)(values)
    expected_error = 0.0
    for slot, count in zip(
        np.asarray(transfer.target_slots),
        np.asarray(transfer.target_dof_count),
        strict=True,
    ):
        element = target.discretization.elements[0][0]
        local_nodes = np.asarray(element.reference_nodes)[:count]
        lower = np.asarray(target.geometry.reference_lower)[slot]
        upper = np.asarray(target.geometry.reference_upper)[slot]
        global_nodes = lower + local_nodes * (upper - lower)
        expected = global_nodes[:, 0] ** min(order, 2) + global_nodes[:, 1]
        expected_error = max(
            expected_error,
            float(np.max(np.abs(np.asarray(transferred[slot, :count]) - expected))),
        )
    passed = bool(geometry_evidence.passed and expected_error <= 2.0e-11)
    return {
        "order": order,
        "active_cells": refined.topology.active_count,
        "degree_buckets": int(np.count_nonzero(np.asarray(target.worksets.bucket_valid))),
        "preparation_seconds": preparation_seconds,
        "geometry_coverage_error": float(geometry_evidence.child_coverage_error),
        "interface_coordinate_error": float(geometry_evidence.interface_coordinate_error),
        "transfer_polynomial_error": expected_error,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/hp_spectral_element.json"),
    )
    args = parser.parse_args()
    results = [benchmark(order) for order in (2, 3, 5)]
    if not all(bool(value["passed"]) for value in results):
        raise RuntimeError("Adaptive hp benchmark qualification failed.")
    args.output.write_text(json.dumps({"quadrilateral": results}, indent=2) + "\n")
    print(json.dumps({"quadrilateral": results}, indent=2))


if __name__ == "__main__":
    main()
