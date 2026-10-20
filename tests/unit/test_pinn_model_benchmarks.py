import math

import jax.numpy as jnp
import pytest

from tools.pinn_model_benchmarks import (
    _manufactured_solution,
    _pde_residual,
    _SCENARIOS,
    PINNBenchmarkRecord,
    PINNBenchmarkScenario,
    run_multifidelity_pinn_benchmark,
    run_pinn_model_benchmark,
    run_pinn_model_benchmarks,
)


_SMOKE_POISSON = PINNBenchmarkScenario(
    "smoke-poisson",
    "poisson",
    frequency=1,
    train_points=6,
    evaluation_points=9,
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
        record = run_pinn_model_benchmark(
            architecture,
            _SMOKE_POISSON,
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
        assert record.dtype == "float64"
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


def test_explicit_float32_mlp_workload_stays_float32() -> None:
    def record(dtype: str | None) -> PINNBenchmarkRecord:
        return run_pinn_model_benchmark(
            "mlp",
            _SMOKE_POISSON,
            seed=0,
            width=4,
            depth=1,
            steps=1,
            learning_rate=1e-3,
            dtype=dtype,
        )

    default = record(None)
    explicit = record("float32")

    assert default.dtype == "float64"
    # The evaluation grid, training points and parameters share the workload
    # dtype, so prediction and derivatives are not promoted to float64.
    assert explicit.dtype == "float32"
    assert explicit.parameter_count == default.parameter_count
    assert len(explicit.losses) == 1
    assert math.isfinite(explicit.final_loss)
    assert math.isfinite(explicit.relative_l2)
    assert math.isfinite(explicit.relative_h1)


@pytest.mark.parametrize("architecture", ("modified_mlp", "piratenet", "siren"))
def test_explicit_dtype_refuses_architectures_without_dtype_construction(
    architecture: str,
) -> None:
    with pytest.raises(ValueError, match="supported only for architectures"):
        run_pinn_model_benchmarks(
            architectures=("mlp", architecture),
            scenarios=("poisson-smooth",),
            seeds=(0,),
            steps=0,
            dtype="float32",
        )
