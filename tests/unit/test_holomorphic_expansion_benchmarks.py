#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from tools.holomorphic_expansion_benchmarks import (
    run_holomorphic_expansion_benchmarks,
)


def test_holomorphic_expansion_benchmark_contract():
    result = run_holomorphic_expansion_benchmarks()
    assert result["kind"] == "holomorphic-expansion-benchmark"
    assert result["passed"]
    assert result["benchmark_id"]
    assert result["constraint_operator"]["rank"] == 3
    assert result["constraint_operator"]["nullity"] == 9
    assert result["constraint_operator"]["target_residual"] < 1e-10
    assert result["projection_residual"] < 1e-10
    assert result["conditional_boundary_residual"] < 1e-10
    assert result["physical_functional_error"] < 1e-10
    assert result["continuous_trace_kind"] == "continuous-subspace-exact"
    assert result["continuous_trace_error"] < 1e-10
    assert result["multijet_error"] < 1e-9
    assert result["pluriharmonic_laplacian_residual"] < 1e-9
    assert result["domain_holomorphic_certificate"]
