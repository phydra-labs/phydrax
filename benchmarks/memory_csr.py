"""CSR preparation retained-memory scaling (small default, no large execution)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

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
    logical_array_bytes,
)
from phydrax.applications import accelerator
from phydrax.discretization import TensorGridPlan, UniformCellAxisSpec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/memory_csr.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    scale = phx.ElectromagneticScaleContract.si()
    rest = float(scale.electron_mass) * float(scale.speed_of_light) ** 2
    rows: list[dict[str, object]] = []
    capacities = (16, 32, 64) if args.smoke else (64, 128, 256, 512, 1024, 2048, 4096)
    for columns in capacities:

        def prepare() -> accelerator.CSRPlan:
            grid = TensorGridPlan(
                (
                    UniformCellAxisSpec(columns // 2),
                    UniformCellAxisSpec(2),
                    UniformCellAxisSpec(9),
                )
            ).prepare(
                np.asarray([[-1e-4, -1e-4, -1e-3], [1e-4, 1e-4, 1e-3]], dtype=np.float64)
            )
            lattice = accelerator.CSRLattice(
                np.full(min(columns, 256), 0.1),
                np.full(min(columns, 256), 0.1),
                element_ids=tuple(f"e{i}" for i in range(min(columns, 256))),
            )
            return accelerator.CSRPlan(
                "3d-retarded-mesh",
                lattice,
                scale,
                grid,
                reference_rest_energy=rest,
                reference_momentum=rest * np.sqrt(1e8 - 1),
                capacity=4,
                far_nodes=4,
                resources=accelerator.CSRResources(
                    observer_chunk=4,
                    maximum_retarded_pairs=2**32,
                    maximum_kernel_bytes=2**36,
                ),
            )

        start = time.perf_counter()
        plan, peak = host_allocation_peak(prepare)
        rows.append(
            {
                "capacity": columns,
                "preparation_seconds": time.perf_counter() - start,
                "host_peak_bytes": peak,
                "retained_bytes": logical_array_bytes(plan),
                "row_workspace_bytes": 4 * min(4, columns) * columns,
                "estimate": plan.estimate._asdict(),
            }
        )
    root = Path(__file__).resolve().parents[1]
    write_json_atomic(
        Path(args.output),
        {
            "identity": capture_benchmark_identity(
                root, Path(__file__), ("retained_bytes", "row_workspace_bytes")
            ).to_dict(),
            "environment": capture_environment().to_dict(),
            "rows": rows,
            "row_workspace_exponent": fitted_scaling_exponent(
                capacities, tuple(16 * columns for columns in capacities)
            ),
            "process_peak_rss_bytes": process_peak_rss_bytes(),
        },
    )


if __name__ == "__main__":
    main()
