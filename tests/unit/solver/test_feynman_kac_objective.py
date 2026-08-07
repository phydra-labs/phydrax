import jax.numpy as jnp
import optax
import pytest

import phydrax as phx
from phydrax.objectives._feynman_kac import FeynmanKacRegressionObjective
from phydrax.stochastic._bsde import BSDEPathBatch, BSDEProblem
from phydrax.stochastic._feynman_kac import (
    FeynmanKacLabelBatch,
    FeynmanKacSamplingPlan,
)


def _problem():
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


def _plan(*, refresh_mode="fixed", control=False):
    return FeynmanKacSamplingPlan(
        terminal_time=1.0,
        sampling_mode="queries",
        num_paths_per_query=8,
        num_time_steps=2,
        control_target_mode="martingale" if control else "none",
        refresh_mode=refresh_mode,
    )


def _labels(problem, plan, *, valid=None, control=False):
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
        valid=jnp.ones((3,), dtype=bool) if valid is None else valid,
        control_valid=(jnp.asarray([True, True, False]) if control else None),
        sample_weights=jnp.asarray([1.0, 2.0, 1.0]),
        source_path_count=8,
    )


def test_fixed_regression_objective_trains_a_global_value_parameter():
    problem = _problem()
    plan = _plan()
    labels = _labels(problem, plan)
    objective = FeynmanKacRegressionObjective(
        problem,
        plan,
        value_name="value",
        labels=labels,
    )
    domain = phx.domain.Interval1d(0.0, 1.0)
    solver = phx.solver.FunctionalSolver(
        functions={"value": domain.Parameter(jnp.asarray([0.0]))},
        constraints=(),
        objectives=(objective,),
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


def test_resampled_provider_is_called_once_per_optimizer_update():
    problem = _problem()
    plan = _plan(refresh_mode="resample")
    labels = _labels(problem, plan)
    calls = []

    def provider(key):
        calls.append(key)
        return labels

    objective = FeynmanKacRegressionObjective(
        problem,
        plan,
        value_name="value",
        labels=provider,
    )
    domain = phx.domain.Interval1d(0.0, 1.0)
    phx.solver.FunctionalSolver(
        functions={"value": domain.Parameter(jnp.asarray([0.0]))},
        constraints=(),
        objectives=(objective,),
    ).solve(
        num_iter=5,
        optim=optax.sgd(0.1),
        jit=True,
        keep_best=False,
        log_every=0,
    )

    assert len(calls) == 5


def test_control_targets_can_train_against_value_autodiff():
    problem = _problem()
    plan = _plan(control=True)
    labels = _labels(problem, plan, control=True)
    objective = FeynmanKacRegressionObjective(
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


def test_zero_valid_mass_and_provenance_mismatch_fail_early():
    problem = _problem()
    plan = _plan()
    invalid = _labels(problem, plan, valid=jnp.zeros((3,), dtype=bool))
    objective = FeynmanKacRegressionObjective(
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
        FeynmanKacRegressionObjective(
            problem,
            other_plan,
            value_name="value",
            labels=_labels(problem, plan),
        )
