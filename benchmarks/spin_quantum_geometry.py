#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

from benchmarks._runtime import capture_environment, logical_array_bytes, measure_host
from phydrax.applications import spin_foam, spin_network
from phydrax.operators.quantum.lattice import SU2SectorResourcePolicy


def benchmark_case(twice_spin: int, quadrature_order: int):
    resources = SU2SectorResourcePolicy(
        maximum_product_dimension=100_000,
        maximum_sector_dimension=10_000,
        maximum_matrix_elements=100_000_000,
    )
    edges = tuple(
        spin_network.SpinNetworkEdge(f"e{index}", "left", "right", twice_spin)
        for index in range(3)
    )
    network, network_seconds = measure_host(
        lambda: spin_network.prepare_spin_network(
            spin_network.SpinNetworkGraphPlan(
                ("left", "right"),
                edges,
                {
                    "left": ("e0", "e1", "e2"),
                    "right": ("e0", "e1", "e2"),
                },
                resources,
            ),
            immirzi_parameter=0.2,
            planck_length_squared=1.0,
        )
    )
    booster_plan, booster_plan_seconds = measure_host(
        lambda: spin_foam.SL2CBoosterReferencePlan(
            radial_cutoff=20.0,
            quadrature_order=quadrature_order,
            tolerance=1e-10,
        )
    )
    booster, booster_seconds = measure_host(
        lambda: spin_foam.evaluate_zero_spin_b4_booster(booster_plan)
    )
    return {
        "axes": {
            "edge_twice_spin": twice_spin,
            "vertex_multiplicity": int(network.vertex_multiplicities[0]),
            "spin_network_dimension": network.basis_dimension,
            "booster_quadrature_order": quadrature_order,
        },
        "ids": {
            "network": network.prepared_id,
            "booster": booster_plan.plan_id,
        },
        "host_seconds": {
            "spin_network_prepare": network_seconds,
            "booster_plan": booster_plan_seconds,
            "booster_execute": booster_seconds,
        },
        "logical_bytes": {
            "network": logical_array_bytes(network),
            "booster": logical_array_bytes(booster),
        },
        "scientific_residuals": {
            "network_orthonormality": float(
                max(network.evidence.vertex_orthonormality_residuals)
            ),
            "booster_relative_error": float(booster.relative_error),
            "booster_tail_bound": float(booster.cutoff_tail_upper_bound),
        },
        "successful": bool(network.evidence.accepted and booster.accepted),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--twice-spins", nargs="+", type=int, default=(2, 4))
    parser.add_argument("--quadrature-order", type=int, default=256)
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    if (
        any(value < 0 for value in arguments.twice_spins)
        or arguments.quadrature_order < 16
    ):
        raise ValueError("Spin/booster benchmark axes are invalid.")
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": [
            benchmark_case(value, arguments.quadrature_order)
            for value in arguments.twice_spins
        ],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
