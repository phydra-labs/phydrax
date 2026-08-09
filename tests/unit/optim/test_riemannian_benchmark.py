#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from tools.riemannian_optim_benchmarks import (
    run_qualification_benchmarks,
    run_smoke_benchmarks,
)


def test_riemannian_benchmark_records_progress_and_invariants():
    report = run_smoke_benchmarks()
    line_search_records = {
        record["name"]: record for record in report["line_search_records"]
    }
    adaptive_records = {
        record["name"]: record for record in report["adaptive_records"]
    }
    records = {record["name"]: record for record in report["records"]}

    assert set(records) == {
        "affine_invariant_spd",
        "mixed_pytree",
        "special_orthogonal",
        "sphere",
        "stiefel",
    }
    for record in records.values():
        assert record["final_objective"] < record["initial_objective"]
        assert record["constraint_residual_max"] < 1e-8
        assert record["first_step_seconds"] > 0.0
        assert record["steady_step_seconds"] > 0.0
        assert record["output_bytes"] > 0
        assert record["parameter_count"] > 0
        assert record["num_manifold_leaves"] == 1

    assert records["sphere"]["unit_norm_error"] < 1e-10
    assert records["stiefel"]["orthogonality_error"] < 1e-10
    assert records["special_orthogonal"]["orthogonality_error"] < 1e-10
    assert records["special_orthogonal"]["determinant"] > 0.0

    assert set(adaptive_records) == set(records)
    for record in adaptive_records.values():
        assert record["optimizer"] == "riemannian_adam"
        assert record["final_objective"] < record["initial_objective"]
        assert record["constraint_residual_max"] < 1e-8
        assert record["adaptive_denominator_minimum"] > 0.0
        assert (
            record["adaptive_denominator_maximum"]
            >= record["adaptive_denominator_minimum"]
        )
    assert set(line_search_records) == {
        "riemannian_conjugate_gradient",
        "riemannian_lbfgs",
    }
    for record in line_search_records.values():
        assert record["final_objective"] < record["initial_objective"]
        assert record["line_search_evaluations"] >= 1
        assert record["line_search_accepted"]
        assert record["constraint_residual_max"] < 1e-8
        assert record["first_step_seconds"] > 0.0
        assert record["steady_step_seconds"] > 0.0
        assert record["transport_first_seconds"] > 0.0
        assert record["transport_steady_seconds"] > 0.0
    assert records["affine_invariant_spd"]["minimum_eigenvalue"] > 0.0


def test_advanced_optimizers_qualify_across_builtin_manifold_families():
    report = run_qualification_benchmarks(steps=8)
    records = {
        (record["geometry"], record["optimizer"]): record
        for record in report["records"]
    }

    assert {geometry for geometry, _ in records} == {
        "affine_invariant_spd",
        "fixed_rank",
        "grassmann",
        "hyperboloid",
        "oblique",
        "poincare_ball",
        "probability_simplex",
        "special_orthogonal",
        "sphere",
        "stiefel",
    }
    assert {optimizer for _, optimizer in records} == {
        "riemannian_conjugate_gradient",
        "riemannian_lbfgs",
    }
    for record in records.values():
        assert record["final_objective"] < record["initial_objective"]
        assert record["constraint_residual_max"] < 1e-6
        assert record["accepted_step_count"] > 0
