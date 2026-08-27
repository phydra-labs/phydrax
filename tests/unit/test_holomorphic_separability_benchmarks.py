#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from tools.holomorphic_separability_benchmarks import (
    run_holomorphic_separability_benchmarks,
)


def test_holomorphic_separability_benchmark_schema_and_guarantees():
    result = run_holomorphic_separability_benchmarks()
    assert result["kind"] == "holomorphic-separability-benchmark"
    assert result["passed"]
    assert result["benchmark_id"]
    assert set(result["parameters"]) == {
        "dense_hmlp",
        "factorized_hmlp",
        "product_potential",
        "branch_bundle",
        "constrained_polynomial",
    }
    assert result["parameters"]["factorized_hmlp"] < result["parameters"]["dense_hmlp"]
    assert max(result["cauchy_riemann_residuals"].values()) < 1e-10
    assert result["product_jet_error"] < 1e-10
    assert result["laplace_residual"] < 1e-10
    assert result["constrained_laplace_residual"] < 1e-10
    assert (
        result["constraint_evidence"]["residual"]
        <= result["constraint_evidence"]["tolerance"]
    )
    assert result["constraint_evidence"]["rank"] == 2
    assert result["constraint_evidence"]["nullity"] == 6
    assert result["constraint_evidence"]["preparation_seconds"] >= 0.0
    assert result["factor_gauge_imbalance"] >= 1.0
