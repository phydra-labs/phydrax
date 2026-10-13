"""Host preparation scaling and sparse action compiler evidence."""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._memory import (
    fitted_scaling_exponent,
    host_allocation_peak,
    process_peak_rss_bytes,
)
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


def make_plan(
    cells: int, groups: int
) -> phx.applications.reactor_physics.MultigroupDiffusionPlan:
    payload = b"reactor-memory-data"
    reference = phx.qualification.ReferenceArtifactManifest(
        "reactor-memory-data",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="fixture",
        nondimensionalization={"identity": 1.0},
        uncertainty=None,
        lineage_ids=("synthetic",),
    )
    provenance = phx.nuclear.NuclearDataProvenance(
        reference, "https://example.invalid/reactor", "synthetic", "release", "multigroup"
    )
    energy = phx.nuclear.EnergyGroupStructure(
        np.arange(groups + 1, dtype=np.float64),
        phx.units.MEGAELECTRONVOLT,
        source_id="memory-groups",
    )
    geometry = phx.discretization.finite_volume.MetricLinePlan(
        np.arange(cells + 1, dtype=np.float64),
        np.ones(cells, dtype=np.float64),
        np.ones(cells + 1, dtype=np.float64),
        "memory-slab",
    )
    scattering = np.full((cells, groups, groups), 0.01, dtype=np.float64)
    scattering[:, np.arange(groups), np.arange(groups)] = 0.0
    material = phx.applications.reactor_physics.MultigroupMaterialData(
        energy,
        np.ones((cells, groups), dtype=np.float64),
        np.ones((cells, groups), dtype=np.float64),
        scattering,
        np.ones((cells, groups), dtype=np.float64),
        np.full((cells, groups), 1.0 / groups, dtype=np.float64),
        provenance,
        "memory-material",
    )
    return phx.applications.reactor_physics.MultigroupDiffusionPlan(geometry, material)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/memory_reactor_diffusion.json")
    )
    args = parser.parse_args()
    capacities = [8, 16, 32] if args.smoke else [64, 128, 256, 512]
    records: list[dict[str, Any]] = []
    for cells in capacities:
        plan = make_plan(cells, 4)
        plan.prepare()  # Exclude one-time substrate initialization from host measurement.
        started = time.perf_counter()
        prepared, peak = host_allocation_peak(plan.prepare)
        preparation_seconds = time.perf_counter() - started
        action = jax.jit(lambda x: prepared.loss_operator.mv(x))
        x = jnp.ones(cells * 4, dtype=jnp.float64)
        compiled, timing = measure_lower_and_compile(
            lambda: action.lower(x),
            lambda lowered: lowered.compile(),
        )
        _, warm = measure_repeated(lambda: compiled(x), warmup=1, repeats=3)
        evidence = compiler_evidence(
            compiled.cost_analysis(),
            compiled.memory_analysis(),
            source="jax-compiled",
        )
        records.append(
            {
                "cells": cells,
                "groups": 4,
                "host_peak_bytes": peak,
                "preparation_seconds": preparation_seconds,
                "retained_bytes": logical_array_bytes(prepared),
                "dense_workspace_bytes": 16 * (cells * 4) ** 2,
                "sparse_capacity": cells * 4 * 4 + 2 * (cells - 1) * 4,
                "plan_id": prepared.plan_id,
                "lowering_seconds": timing.lowering_seconds,
                "compilation_seconds": timing.compilation_seconds,
                "warm": warm.to_dict(),
                "compiler": asdict(evidence),
            }
        )
    result = {
        "records": records,
        "host_scaling_exponent": fitted_scaling_exponent(
            capacities, [r["host_peak_bytes"] for r in records]
        ),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "environment": capture_environment().to_dict(),
        "identity": capture_benchmark_identity(
            Path.cwd(), Path(__file__), records[0].keys()
        ).to_dict(),
    }
    write_json_atomic(args.output, result)
    print(
        {
            "host_peak_bytes": [r["host_peak_bytes"] for r in records],
            "exponent": result["host_scaling_exponent"],
        }
    )


if __name__ == "__main__":
    main()
