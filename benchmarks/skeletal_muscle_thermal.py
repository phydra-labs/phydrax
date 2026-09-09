#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import numpy as np

from benchmarks._runtime import capture_environment
from tools.skeletal_muscle_thermal_qualification import (
    manufactured_case,
    manufactured_source,
)


def _case(resolution):
    start = time.perf_counter()
    prepared = manufactured_case(resolution, heterogeneous=True)
    prepare_ms = 1000 * (time.perf_counter() - start)
    state = prepared.initial_state()
    source = manufactured_source(prepared, state)
    action = eqx.filter_jit(prepared.propose)
    start = time.perf_counter()
    candidate = action(state, source)
    jax.block_until_ready(candidate)
    compile_and_first_ms = 1000 * (time.perf_counter() - start)
    elapsed = []
    for _ in range(5):
        start = time.perf_counter()
        candidate = action(state, source)
        jax.block_until_ready(candidate)
        elapsed.append(1000 * (time.perf_counter() - start))
    retained = prepared.plan.projection.operator
    source_action = eqx.filter_jit(retained.mv)
    jax.block_until_ready(source_action(source.retained_power_W))
    start = time.perf_counter()
    jax.block_until_ready(source_action(source.retained_power_W))
    projection_ms = 1000 * (time.perf_counter() - start)
    return {
        "resolution": resolution,
        "cell_count": int(prepared.geometry.cell_volume_m3.size),
        "temperature_dofs": int(state.temperature_K.size),
        "source_map_edges": int(retained.coefficients.size),
        "finite_element_source_edges": int(
            prepared.geometry.source_load.coefficients.size
        ),
        "prepared_array_leaf_bytes_including_shared_references": sum(
            int(x.size * x.dtype.itemsize)
            for x in jax.tree.leaves(prepared)
            if eqx.is_array(x)
        ),
        "prepare_ms": prepare_ms,
        "compile_and_first_ms": compile_and_first_ms,
        "median_candidate_ms": float(np.median(elapsed)),
        "source_projection_ms": projection_ms,
        "linear_iterations": int(candidate.evidence.linear_iterations),
        "linear_residual_norm": float(candidate.evidence.linear_residual_norm),
        "balance_residual_J": float(candidate.ledger.balance_residual_J),
        "successful": bool(candidate.evidence.successful),
        "prepared_id": prepared.prepared_id,
        "dtype": str(state.temperature_K.dtype),
        "execution": "local-unsharded native FEM/PCG; no physical multi-device claim",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    cases = [_case(1)] if args.smoke else [_case(n) for n in (2, 4, 8)]
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "all_successful": all(x["successful"] for x in cases),
        "claim_scope": "Manufactured numerical scalar Pennes only",
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
