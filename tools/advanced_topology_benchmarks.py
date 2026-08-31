#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import time

import jax.numpy as jnp

import phydrax as phx


def timed(function):
    started = time.perf_counter()
    value = function()
    return value, time.perf_counter() - started


def main():
    topology = phx.geometry.simplicial.TriangleTopology(
        jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
        num_vertices=3,
    ).cell_complex_topology()
    complex = phx.topology.CellSubcomplex.full(topology)
    identity = phx.topology.CellularChainMap.identity(complex)
    filtration = phx.topology.CellFiltration(
        complex,
        (
            jnp.asarray([0.0, 0.5, 1.0]),
            jnp.asarray([0.5, 1.0, 1.0]),
            jnp.asarray([2.0]),
        ),
        source_id="advanced-benchmark",
    )
    cone, cone_seconds = timed(
        lambda: phx.topology.compute_mapping_cone_homology(
            identity,
            coefficients=phx.topology.PrimeField(2),
        )
    )
    extended, extended_seconds = timed(
        lambda: phx.topology.compute_extended_persistence(
            filtration,
            coefficients=phx.topology.PrimeField(2),
        )
    )
    integral, integral_seconds = timed(
        lambda: phx.topology.compute_integral_homology(complex)
    )
    result = {
        "cone_acyclic": cone.acyclic,
        "cone_seconds": cone_seconds,
        "extended_seconds": extended_seconds,
        "extended_intervals": sum(
            value.interval_count
            for value in (
                extended.ordinary,
                extended.relative,
                extended.extended_positive,
                extended.extended_negative,
            )
        ),
        "integral_seconds": integral_seconds,
        "integral_free_ranks": [value.free_rank for value in integral.degrees],
        "integral_torsion_counts": [
            len(value.torsion_invariants) for value in integral.degrees
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
