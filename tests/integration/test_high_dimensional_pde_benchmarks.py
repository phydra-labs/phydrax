import jax
import jax.numpy as jnp

from tools.high_dimensional_pde_benchmarks import (
    HIGH_DIMENSIONAL_METHOD_MATRIX,
    linear_hjb_gradient,
    linear_hjb_value,
    ornstein_uhlenbeck_score,
    quadratic_heat_gradient,
    quadratic_heat_value,
    quartic_field,
    quartic_laplacian,
    run_high_dimensional_method_benchmarks,
    run_high_dimensional_reference_benchmarks,
)


def test_high_dimensional_reference_benchmark_schema_and_replay():
    first = run_high_dimensional_reference_benchmarks(
        (4, 12), num_samples=2048, repeats=1, seed=12
    )
    second = run_high_dimensional_reference_benchmarks(
        (4, 12), num_samples=2048, repeats=1, seed=12
    )

    assert first["schema_version"] == 1
    assert first["dimensions"] == [4, 12]
    assert first["passed"]
    assert len(first["records"]) == 8
    for left, right in zip(first["records"], second["records"], strict=True):
        assert left["problem_id"] == right["problem_id"]
        assert left["method"] == right["method"]
        assert left["value"] == right["value"]
        assert left["reference"] == right["reference"]
        assert left["absolute_error"] == right["absolute_error"]
        assert left["reported_standard_error"] == right["reported_standard_error"]
        assert left["passed"] == right["passed"]


def test_high_dimensional_method_matrix_runs_with_common_result_schema():
    result = run_high_dimensional_method_benchmarks(
        (4, 12),
        num_samples=2048,
        num_probes=64,
        repeats=1,
        seed=13,
    )

    assert result["schema_version"] == 2
    assert result["dimensions"] == [4, 12]
    assert result["passed"]
    assert len(result["method_matrix"]) == len(HIGH_DIMENSIONAL_METHOD_MATRIX)
    assert len(result["records"]) == 8
    assert {record["method"] for record in result["records"]} == {
        "query-feynman-kac",
        "hutchinson-trace",
        "dimension-sampling",
        "implicit-score-matching",
    }
    for record in result["records"]:
        assert record["target_type"] in {
            "point-value",
            "differential-operator",
            "score-field",
        }
        assert record["compile_ms"] >= 0.0
        assert record["mean_wall_ms"] >= 0.0
        assert record["working_set_bytes"] > 0
        assert record["status"] == "completed"
        assert record["success"]
    feynman_kac_records = [
        record
        for record in result["records"]
        if record["method"] == "query-feynman-kac"
    ]
    assert all(record["control_error"] is not None for record in feynman_kac_records)
    assert all(
        record["control_standard_error"] is not None
        for record in feynman_kac_records
    )
    assert all(record["total_wall_ms"] > 0.0 for record in feynman_kac_records)
    score_records = [
        record
        for record in result["records"]
        if record["method"] == "implicit-score-matching"
    ]
    assert all(record["gradient_error"] == 0.0 for record in score_records)
    assert all(record["valid_fraction"] == 1.0 for record in score_records)
    assert all(record["num_samples"] == 128 for record in score_records)


def test_neural_stochastic_method_benchmarks_train_declared_solution_objects():
    result = run_high_dimensional_method_benchmarks(
        (10,),
        num_samples=256,
        score_samples=32,
        num_probes=8,
        repeats=1,
        seed=0,
        include_training=True,
        deep_picard_paths=16,
        deep_picard_queries=8,
        deep_picard_iterations=80,
        deep_bsde_paths=16,
        deep_bsde_time_steps=4,
        deep_bsde_iterations=120,
        deep_splitting_paths=16,
        deep_splitting_time_steps=4,
        deep_splitting_iterations=80,
    )

    training_records = {
        record["method"]: record
        for record in result["records"]
        if record["method"] in {"deep-picard", "deep-bsde", "deep-splitting"}
    }
    deep_picard = training_records["deep-picard"]
    deep_bsde = training_records["deep-bsde"]
    deep_splitting = training_records["deep-splitting"]
    assert result["training_methods_included"]
    assert result["passed"]
    assert set(training_records) == {"deep-picard", "deep-bsde", "deep-splitting"}
    assert all(record["passed"] for record in training_records.values())
    assert all(
        record["absolute_error"] < record["acceptance_tolerance"]
        for record in training_records.values()
    )
    assert deep_picard["gradient_error"] == 0.0
    assert deep_picard["terminal_error"] == 0.0
    assert deep_bsde["control_error"] < deep_bsde["acceptance_tolerance"]
    assert deep_bsde["terminal_error"] < deep_bsde["acceptance_tolerance"]
    assert deep_splitting["gradient_error"] == 0.0
    assert deep_splitting["terminal_error"] == 0.0
    assert all(record["total_wall_ms"] > 0.0 for record in training_records.values())


def test_reference_gradients_and_laplacian_match_autodiff():
    state = jnp.linspace(-1.0, 1.0, 9)

    assert jnp.allclose(
        quadratic_heat_gradient(state),
        jax.grad(lambda value: quadratic_heat_value(0.0, value))(state),
    )
    assert jnp.allclose(
        linear_hjb_gradient(state),
        jax.grad(lambda value: linear_hjb_value(0.0, value))(state),
    )

    hessian = jax.hessian(quartic_field)(state)
    assert jnp.allclose(quartic_laplacian(state), jnp.trace(hessian))


def test_reference_value_and_score_shapes():
    states = jnp.ones((7, 16))
    times = jnp.linspace(0.0, 0.75, 7)

    assert quadratic_heat_value(times, states).shape == (7,)
    assert linear_hjb_value(times, states).shape == (7,)
    assert ornstein_uhlenbeck_score(times, states).shape == states.shape
