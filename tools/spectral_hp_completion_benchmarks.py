#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp

import phydrax as phx


def run() -> dict[str, object]:
    start = perf_counter()
    mesh = phx.discretization.CellMesh(
        jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (1.0, 1.0, 1.0),
                (0.0, 1.0, 1.0),
            ),
            dtype=jnp.float64,
        ),
        (
            phx.discretization.CellBlock(
                "hexahedra",
                "hexahedron",
                jnp.arange(8, dtype=jnp.int32)[None, :],
            ),
        ),
    )
    de_rham = phx.discretization.fem.FiniteElementDeRhamComplex(
        mesh,
        family="tensor-trimmed",
        order=5,
        twist="untwisted",
    )
    de_rham_seconds = perf_counter() - start
    counts = de_rham.cell_counts
    scalar = jnp.arange(counts[0], dtype=jnp.float64)
    circulation = jnp.arange(counts[1], dtype=jnp.float64)
    curl_gradient_defect = float(
        jnp.max(
            jnp.abs(
                de_rham.exterior_derivative(1, de_rham.exterior_derivative(0, scalar))
            ),
            initial=0.0,
        )
    )
    divergence_curl_defect = float(
        jnp.max(
            jnp.abs(
                de_rham.exterior_derivative(
                    2, de_rham.exterior_derivative(1, circulation)
                )
            ),
            initial=0.0,
        )
    )
    marker = phx.solver.RelaxedHPMarking(3, 0.1)
    weights = marker.weights(
        jnp.asarray((1.0, 4.0, 2.0, 3.0)), jnp.ones((4,), dtype=jnp.bool_)
    )
    result = {
        "de_rham": {
            "degree": 5,
            "gradient_shape": (counts[1], counts[0]),
            "curl_gradient_defect": curl_gradient_defect,
            "divergence_curl_defect": divergence_curl_defect,
            "construction_seconds": de_rham_seconds,
        },
        "relaxed_marking": {
            "budget": 3,
            "weight_sum": float(jnp.sum(weights)),
        },
    }
    result["passed"] = bool(
        curl_gradient_defect <= 1.0e-12
        and divergence_curl_defect <= 1.0e-12
        and float(jnp.sum(weights)) <= 3.0 + 1.0e-12
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/spectral_hp_completion.json"),
    )
    args = parser.parse_args()
    result = run()
    if not result["passed"]:
        raise RuntimeError("Spectral hp completion benchmark failed.")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
