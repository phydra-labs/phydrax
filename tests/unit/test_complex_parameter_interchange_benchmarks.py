#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from tools.complex_parameter_interchange_benchmarks import (
    run_complex_parameter_interchange_benchmarks,
)


def test_complex_parameter_interchange_benchmark_contract():
    result = run_complex_parameter_interchange_benchmarks()
    assert result["kind"] == "complex-parameter-interchange-benchmark"
    assert result["passed"]
    assert result["benchmark_id"]
    assert result["model_state_id"]
    assert result["model_error"] == 0.0
    assert result["polynomial_error"] == 0.0
    assert result["constrained_coefficient_error"] < 1e-11
    assert result["constrained_residual"] < 1e-11
    assert result["meromorphic_coefficient_error"] < 1e-11
    assert result["internal_complex_trainable_leaves"] == 0
    assert result["state_payload_bytes"] > 0
    assert result["export_seconds"] >= 0.0
    assert result["import_seconds"] >= 0.0
