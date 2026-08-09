import math

import jax.numpy as jnp
import pytest

from tools.pinn_model_benchmarks import (
    _manufactured_solution,
    _pde_residual,
    _SCENARIOS,
    PINNBenchmarkScenario,
    run_pinn_model_benchmark,
)


@pytest.mark.parametrize(
    "scenario_name",
    ("poisson-smooth", "helmholtz-oscillatory", "allen-cahn-nonlinear"),
)
def test_manufactured_solution_exactly_satisfies_each_pde(scenario_name):
    scenario = _SCENARIOS[scenario_name]
    points = jnp.linspace(-1.0, 1.0, 17)[:, None]
    truth = _manufactured_solution(points, scenario.frequency)
    omega = float(scenario.frequency) * jnp.pi
    residual = _pde_residual(truth, -(omega**2) * truth, points, scenario)

    assert jnp.allclose(residual, 0.0, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("architecture", ("mlp", "modified_mlp", "piratenet", "siren"))
def test_pointwise_benchmark_smoke_covers_every_model_family(architecture):
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
