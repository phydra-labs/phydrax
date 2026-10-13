#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Compile-only linear-memory certification scaling; --smoke runs three sizes."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from phydrax.linalg import banded_eigenvalue_brackets, banded_singular_value_certificate

from ._io import write_json_atomic
from ._memory import fitted_scaling_exponent, host_allocation_peak, process_peak_rss_bytes
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    args = parser.parse_args()
    sizes = args.sizes or ([128, 256, 512] if args.smoke else [512, 1024, 2048])
    records: list[dict[str, Any]] = []
    temporary: dict[str, list[int]] = {
        "eigenvalue-brackets": [],
        "singular-threshold": [],
    }
    for n in sizes:
        bands = jnp.zeros((5, n), dtype=jnp.float64).at[2].set(4)
        bands = bands.at[3, :-1].set(0.3).at[1, 1:].set(0.3)
        bands = bands.at[4, :-2].set(0.1).at[0, 2:].set(0.1)
        for route in temporary:
            if route == "eigenvalue-brackets":
                entry = jax.jit(
                    partial(banded_eigenvalue_brackets, bandwidth=2, max_steps=8)
                )
            else:
                entry = jax.jit(
                    partial(
                        banded_singular_value_certificate,
                        rows=n,
                        lower_bandwidth=2,
                        upper_bandwidth=2,
                        threshold=1.0,
                    )
                )
            (compiled, timing), host_peak = host_allocation_peak(
                lambda: measure_lower_and_compile(
                    lambda: entry.lower(bands), lambda lowered: lowered.compile()
                )
            )
            evidence = compiler_evidence(
                compiled.cost_analysis(),
                compiled.memory_analysis(),
                source="jax-banded-spectrum",
                unavailable_reason="Compiler memory estimates unavailable.",
            )
            if evidence.temporary_bytes is None:
                raise RuntimeError(
                    "Compiler temporary bytes are required for this benchmark."
                )
            temporary[route].append(evidence.temporary_bytes)
            records.append(
                {
                    "rows": n,
                    "bandwidth": 2,
                    "route": route,
                    "compiler": asdict(evidence),
                    "lowering_seconds": timing.lowering_seconds,
                    "compilation_seconds": timing.compilation_seconds,
                    "host_allocation_peak_bytes": host_peak,
                    "logical_input_bytes": logical_array_bytes(bands),
                    "executed": False,
                }
            )
            jax.clear_caches()
    root = Path(__file__).resolve().parents[1]
    fields = (
        "environment",
        "command",
        "configuration",
        "cases",
        "scaling_exponents",
        "process_peak_rss_bytes",
    )
    record = {
        "benchmark_identity": capture_benchmark_identity(
            root, Path(__file__), fields
        ).to_dict(),
        "environment": capture_environment().to_dict(),
        "command": " ".join(sys.argv),
        "configuration": {
            "sizes": sizes,
            "dtype": "float64",
            "bandwidth": 2,
            "new_capability_no_baseline": True,
            "work": "O(n*b^2)",
            "memory": "O(n*b)",
        },
        "cases": records,
        "scaling_exponents": {
            route: fitted_scaling_exponent(sizes, values)
            for route, values in temporary.items()
        },
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }
    write_json_atomic(args.output, record)
    print(record["scaling_exponents"])


if __name__ == "__main__":
    main()
