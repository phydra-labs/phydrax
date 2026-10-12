"""Component minimum-norm memory and independent pinv oracle scaling."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

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
)
from phydrax.discretization.amr._cut_transfer_actions import ComponentMinimumNorm


def main() -> None:
    sizes = [8, 16, 32, 64]
    baseline: list[dict[str, Any]] = []
    candidate: list[dict[str, Any]] = []
    for count in sizes:
        matrix = sp.block_diag(
            [np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]], dtype=np.float64)] * count,
            format="csr",
        )
        matrix = sp.csr_matrix(matrix)
        reference, peak = host_allocation_peak(
            lambda: np.linalg.pinv(matrix.toarray(), rcond=1.0e-9)
        )
        baseline.append(
            {
                "capacity": count,
                "host_peak_bytes": peak,
                "retained_bytes": reference.nbytes,
            }
        )
        solve, peak = host_allocation_peak(
            lambda: ComponentMinimumNorm(matrix, 1.0e-9, 64 * 1024 * 1024)
        )
        rhs = jnp.arange(matrix.shape[0], dtype=jnp.float64)
        compiled, timing = measure_lower_and_compile(
            lambda: jax.jit(lambda value: solve(value)).lower(rhs),
            lambda lowered: lowered.compile(),
        )
        value = np.asarray(compiled(rhs))
        np.testing.assert_allclose(value, reference @ np.asarray(rhs), atol=1.0e-10)
        candidate.append(
            {
                "capacity": count,
                "host_peak_bytes": peak,
                "retained_bytes": logical_array_bytes(solve),
                "component_shapes": solve.component_shapes,
                "rank": solve.rank,
                "nnz": matrix.nnz,
                "oracle_maximum_error": float(
                    np.max(np.abs(value - reference @ np.asarray(rhs)))
                ),
                "compiler": asdict(
                    compiler_evidence(
                        compiled.cost_analysis(),
                        compiled.memory_analysis(),
                        source="jax-compiled",
                    )
                ),
                "lowering_seconds": timing.lowering_seconds,
                "compilation_seconds": timing.compilation_seconds,
            }
        )
    root = Path(__file__).resolve().parents[1]
    common = {
        "identity": capture_benchmark_identity(
            root, Path(__file__), ("capacity", "host_peak_bytes", "retained_bytes")
        ).to_dict(),
        "environment": capture_environment().to_dict(),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "baseline_kind": "independent-dense-pinv-oracle; not pre-edit full-plan capture",
    }
    write_json_atomic(
        root / "benchmarks/memory_cut_cochain_transfer_baseline.json",
        {
            **common,
            "cases": baseline,
            "retained_exponent": fitted_scaling_exponent(
                sizes, [row["retained_bytes"] for row in baseline]
            ),
        },
    )
    write_json_atomic(
        root / "benchmarks/memory_cut_cochain_transfer.json",
        {
            **common,
            "cases": candidate,
            "retained_exponent": fitted_scaling_exponent(
                sizes, [row["retained_bytes"] for row in candidate]
            ),
        },
    )


if __name__ == "__main__":
    main()
