#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from tools.advanced_potential_benchmarks import run_advanced_potential_benchmarks


def test_advanced_potential_benchmark_schema_and_guarantee_split():
    result = run_advanced_potential_benchmarks(
        panels_per_chart=2,
        quadrature_order=4,
    )
    assert result["schema_version"] == 1
    assert result["passed"]
    assert result["holomorphic"]["laplace_residual"] == 0.0
    assert result["boundary_layer"]["pde_exactness"] == "algebraic"
    assert result["boundary_layer"]["validity_region"] == "off-singular-support"
    assert result["boundary_layer"]["approximation_id"]
