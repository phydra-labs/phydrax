from typing import Any

import jax.numpy as jnp
import optax
import pytest

import phydrax as phx
from phydrax.stochastic._bsde import BSDEPathBatch, BSDEProblem
from phydrax.stochastic._feynman_kac import (
    FeynmanKacLabelBatch,
    FeynmanKacSamplingPlan,
)
from phydrax.terms._feynman_kac import FeynmanKacRegressionTerm


def _problem() -> Any:
    paths = BSDEPathBatch(
        jnp.asarray([0.0, 1.0]),
        jnp.zeros((1, 2, 1)),
        jnp.zeros((1, 1, 1)),
        sample_shape=(1,),
        state_shape=(1,),
        noise_shape=(1,),
        path_id="unused",
        process_id="regression",
    )
    return BSDEProblem(
        lambda key: paths,
        lambda time, state, args: jnp.zeros_like(state),
        lambda time, state, args: jnp.ones((1, 1)),
        lambda time, state, value, control, args: jnp.zeros_like(value),
        lambda state, args: jnp.asarray([state[0]]),
        state_shape=(1,),
        noise_shape=(1,),
        output_shape=(1,),
        problem_id="regression-problem",
        process_id="regression",
    )


def _plan(*, refresh_mode: Any = "fixed", control: Any = False) -> Any:
    return FeynmanKacSamplingPlan(
        terminal_time=1.0,
        sampling_mode="queries",
        num_paths_per_query=8,
        num_time_steps=2,
        control_target_mode="martingale" if control else "none",
        refresh_mode=refresh_mode,
    )


def _labels(problem: Any, plan: Any, *, valid: Any = None, control: Any = False) -> Any:
    times = jnp.asarray([0.0, 0.5, 1.0])
    controls = jnp.ones((3, 1, 1)) if control else None
    return FeynmanKacLabelBatch(
        times,
        jnp.zeros((3, 1)),
        jnp.full((3, 1), 2.0),
        state_shape=problem.state_shape,
        noise_shape=problem.noise_shape,
        output_shape=problem.output_shape,
        problem_id=problem.problem_id,
        process_id=problem.process_id,
        plan_id=plan.plan_id,
        value_standard_errors=jnp.full((3, 1), 0.1),
        control_targets=controls,
        control_standard_errors=(jnp.full((3, 1, 1), 0.2) if control else None),
        valid=jnp.ones((3,), dtype="bool") if valid is None else valid,
        control_valid=(jnp.asarray([True, True, False]) if control else None),
        sample_weights=jnp.asarray([1.0, 2.0, 1.0]),
        source_path_count=8,
    )


def test_feynman_kac_objective_scenario_1() -> None:
    problem = _problem()
    plan = _plan()
    labels = _labels(problem, plan)
    objective = FeynmanKacRegressionTerm(
        problem,
        plan,
        value_name="value",
        labels=labels,
    )
    domain = phx.domain.Interval1d(0.0, 1.0)
    solver = phx.solver.FunctionalSolver(
        functions={"value": domain.Parameter(jnp.asarray([0.0]))},
        terms=(objective,),
    )

    initial = objective.loss(solver.functions, batch=labels)
    trained = solver.solve(
        num_iter=40,
        optim=optax.sgd(0.1),
        jit=True,
        keep_best=False,
        log_every=0,
    )
    final = objective.loss(trained.functions, batch=labels)

    assert initial > 3.9
    assert final < 1e-6
    assert objective.diagnostics(trained.functions, batch=labels).passed
    problem = _problem()
    plan = _plan(control=True)
    labels = _labels(problem, plan, control=True)
    objective = FeynmanKacRegressionTerm(
        problem,
        plan,
        value_name="value",
        labels=labels,
        value_weight=0.0,
        control_weight=1.0,
    )
    domain = phx.domain.Interval1d(-2.0, 2.0) @ phx.domain.TimeInterval(0.0, 1.0)
    value = domain.Function("t", "x")(lambda time, state: jnp.asarray([state[0]]))

    assert jnp.allclose(objective.loss({"value": value}, batch=labels), 0.0)
    problem = _problem()
    plan = _plan()
    invalid = _labels(problem, plan, valid=jnp.zeros((3,), dtype="bool"))
    objective = FeynmanKacRegressionTerm(
        problem,
        plan,
        value_name="value",
        labels=invalid,
    )
    domain = phx.domain.Interval1d(0.0, 1.0)

    with pytest.raises(Exception, match="zero valid"):
        objective.loss({"value": domain.Parameter(jnp.asarray([0.0]))}, batch=invalid)

    other_plan = FeynmanKacSamplingPlan(
        terminal_time=2.0,
        sampling_mode="queries",
        refresh_mode="fixed",
    )
    with pytest.raises(ValueError, match="provenance"):
        FeynmanKacRegressionTerm(
            problem,
            other_plan,
            value_name="value",
            labels=_labels(problem, plan),
        )


def test_resampled_provider_is_called_once_per_optimizer_update() -> None:
    problem = _problem()
    plan = _plan(refresh_mode="resample")
    labels = _labels(problem, plan)
    calls = []

    def provider(key: Any) -> Any:
        calls.append(key)
        return labels

    objective = FeynmanKacRegressionTerm(
        problem,
        plan,
        value_name="value",
        labels=provider,
    )
    domain = phx.domain.Interval1d(0.0, 1.0)
    phx.solver.FunctionalSolver(
        functions={"value": domain.Parameter(jnp.asarray([0.0]))},
        terms=(objective,),
    ).solve(
        num_iter=5,
        optim=optax.sgd(0.1),
        jit=True,
        keep_best=False,
        log_every=0,
    )

    assert len(calls) == 5
