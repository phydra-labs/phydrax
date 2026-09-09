#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from phydrax._fingerprint import canonical_json
from tools.operator_correction_benchmarks import run_quick


def test_quick_operator_correction_benchmark_certifies_original_residuals():
    report = run_quick(size=9, modes=2, warmup=0, repeats=1)

    assert report["benchmark"] == "operator-correction-quick"
    assert set(report["solvers"]) == {
        "native_jacobi",
        "operator_subspace",
        "direct_operator",
    }
    assert report["qualification"]["false_success_count"] == 0
    assert report["qualification"]["all_original_residuals_finite"]
    for record in report["solvers"].values():
        assert record["successful"]
        assert record["original_relative_residual"] < 1e-7
        assert record["steady_solve"]["count"] == 1
    canonical_json(report)
