import jax.numpy as jnp
import optax

import phydrax as phx
from phydrax.objectives._deep_bsde import (
    deep_bsde_rollout,
    DeepBSDEShootingObjective,
)
from phydrax.solver._deep_bsde import solve_deep_bsde
from phydrax.stochastic._bsde import BSDEPathBatch, BSDEProblem


def _brownian_paths():
    increments = jnp.asarray([[[0.2], [-0.1]], [[-0.3], [0.4]]])
    states = jnp.concatenate(
        (jnp.zeros((2, 1, 1)), jnp.cumsum(increments, axis=1)), axis=1
    )
    return BSDEPathBatch(
        jnp.asarray([0.0, 0.5, 1.0]),
        states,
        increments,
        sample_shape=(2,),
        state_shape=(1,),
        noise_shape=(1,),
        path_id="brownian",
        process_id="brownian",
    )


def _problem(paths, *, terminal=None):
    return BSDEProblem(
        lambda key: paths,
        lambda time, state, args: jnp.zeros_like(state),
        lambda time, state, args: jnp.ones((1, 1)),
        lambda time, state, value, control, args: jnp.zeros_like(value),
        (
            (lambda state, args: jnp.asarray([state[0]]))
            if terminal is None
            else terminal
        ),
        state_shape=(1,),
        noise_shape=(1,),
        output_shape=(1,),
        problem_id="brownian-value",
        process_id=paths.process_id,
    )


def _constant(domain, value):
    return phx.domain.DomainFunction(domain=domain, deps=(), func=jnp.asarray(value))


def test_deep_bsde_rollout_reproduces_linear_brownian_solution():
    paths = _brownian_paths()
    problem = _problem(paths)
    domain = phx.domain.Interval1d(-1.0, 1.0)
    initial = _constant(domain, [0.0])
    control = _constant(domain, [[1.0]])

    rollout = deep_bsde_rollout(problem, paths, initial, control)
    objective = DeepBSDEShootingObjective(
        problem,
        initial_value_name="initial",
        control_name="control",
        sampling_mode="fixed",
        fixed_paths=paths,
    )

    assert jnp.allclose(rollout.values[..., 1:, 0], paths.states[..., 1:, 0])
    assert jnp.allclose(rollout.terminal_residual, 0.0)
    assert jnp.allclose(
        objective.loss({"initial": initial, "control": control}, batch=paths), 0.0
    )
    assert objective.diagnostics(
        {"initial": initial, "control": control}, batch=paths
    ).passed


def test_solve_deep_bsde_trains_initial_value_and_removes_temporary_objective():
    paths = BSDEPathBatch(
        jnp.asarray([0.0, 0.5, 1.0]),
        jnp.zeros((8, 3, 1)),
        jnp.zeros((8, 2, 1)),
        sample_shape=(8,),
        state_shape=(1,),
        noise_shape=(1,),
        path_id="deterministic",
        process_id="deterministic",
    )
    problem = _problem(paths, terminal=lambda state, args: jnp.asarray([2.0]))
    domain = phx.domain.Interval1d(-1.0, 1.0)
    solver = phx.solver.FunctionalSolver(
        functions={
            "initial": domain.Parameter(jnp.asarray([0.0])),
            "control": _constant(domain, [[0.0]]),
        },
        constraints=(),
    )

    result = solve_deep_bsde(
        solver,
        problem,
        initial_value_name="initial",
        control_name="control",
        num_iter=30,
        optim=optax.sgd(0.25),
        sampling_mode="fixed",
        fixed_paths=paths,
        validation_paths=paths,
        keep_best=False,
    )

    assert result.diagnostics.terminal_rmse < 1e-8
    assert jnp.allclose(result.diagnostics.initial_mean, jnp.asarray([2.0]), atol=1e-8)
    assert result.diagnostics.passed
    assert result.solver.objectives == solver.objectives
    assert jnp.allclose(solver["initial"].func(), jnp.asarray([0.0]))
