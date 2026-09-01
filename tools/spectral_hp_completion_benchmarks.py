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
    de_rham = phx.discretization.fem.TensorDeRhamComplex(5, 3)
    de_rham_seconds = perf_counter() - start
    marker = phx.solver.RelaxedHPMarking(3, 0.1)
    weights = marker.weights(
        jnp.asarray((1.0, 4.0, 2.0, 3.0)), jnp.ones((4,), dtype=bool)
    )
    result = {
        "de_rham": {
            "degree": 5,
            "gradient_shape": de_rham.gradient.shape,
            "curl_gradient_defect": float(de_rham.grad_curl_defect),
            "divergence_curl_defect": float(de_rham.curl_div_defect),
            "construction_seconds": de_rham_seconds,
        },
        "relaxed_marking": {
            "budget": 3,
            "weight_sum": float(jnp.sum(weights)),
        },
    }
    result["passed"] = bool(
        result["de_rham"]["curl_gradient_defect"] <= 1.0e-12
        and result["de_rham"]["divergence_curl_defect"] <= 1.0e-12
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
