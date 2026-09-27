import math

import jax.numpy as jnp

from tools.pinn_model_benchmarks import (
    _manufactured_solution,
    _pde_residual,
    _SCENARIOS,
    PINNBenchmarkScenario,
    run_multifidelity_pinn_benchmark,
    run_pinn_model_benchmark,
)


def test_pinn_model_benchmarks_scenario_1() -> None:
    for scenario_name in (
        "poisson-smooth",
        "helmholtz-oscillatory",
        "allen-cahn-nonlinear",
    ):
        scenario = _SCENARIOS[scenario_name]
        points = jnp.linspace(-1.0, 1.0, 17)[:, None]
        truth = _manufactured_solution(points, scenario.frequency)
        omega = float(scenario.frequency) * jnp.pi
        residual = _pde_residual(truth, -(omega**2) * truth, points, scenario)

        assert jnp.allclose(residual, 0.0, atol=1e-11, rtol=1e-11)
    for architecture in ("mlp", "modified_mlp", "piratenet", "siren"):
        scenario = PINNBenchmarkScenario(
            "smoke-poisson",
            "poisson",
            frequency=1,
            train_points=6,
            evaluation_points=9,
        )
        record = run_pinn_model_benchmark(
            architecture,
            scenario,
            seed=0,
            width=4,
            depth=1,
            steps=0,
            learning_rate=1e-3,
        )

        assert record.architecture == architecture
        assert record.equation == "poisson"
        assert record.parameter_count > 0
        assert record.target_parameter_count > 0
        assert math.isfinite(record.initial_loss)
        assert math.isfinite(record.final_loss)
        assert math.isfinite(record.relative_l2)
        assert math.isfinite(record.relative_h1)
    result = run_multifidelity_pinn_benchmark(
        seed=0,
        low_steps=1,
        target_steps=1,
    )
    # ty: ignore[invalid-argument-type]
    assert math.isfinite(result["target_rmse"])
    # ty: ignore[invalid-argument-type]
    assert math.isfinite(result["target_physics_loss"])
    assert result["parent_unchanged"]
