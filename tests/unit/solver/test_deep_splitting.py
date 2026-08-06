import jax
import jax.numpy as jnp
import optax

import phydrax as phx
from phydrax.objectives._deep_splitting import (
    deep_splitting_labels,
    DeepSplittingRegressionObjective,
)
from phydrax.solver._deep_splitting import solve_deep_splitting
from phydrax.stochastic._bsde import BSDEPathBatch, BSDEProblem


def _paths():
    return BSDEPathBatch(
        jnp.asarray([0.0, 0.5, 1.0]),
        jnp.zeros((16, 3, 1)),
        jnp.zeros((16, 2, 1)),
        sample_shape=(16,),
        state_shape=(1,),
        noise_shape=(1,),
        path_id="splitting-paths",
        process_id="splitting",
    )


def _problem(paths):
    return BSDEProblem(
        lambda key: paths,
        lambda time, state, args: jnp.zeros_like(state),
        lambda time, state, args: jnp.ones((1, 1)),
        lambda time, state, value, control, args: jnp.asarray([1.0]),
        lambda state, args: jnp.asarray([0.0]),
        state_shape=(1,),
        noise_shape=(1,),
        output_shape=(1,),
        problem_id="constant-source",
        process_id=paths.process_id,
    )


def test_deep_splitting_labels_use_explicit_right_endpoint_source():
    paths = _paths()
    problem = _problem(paths)
    labels = deep_splitting_labels(
        problem,
        paths,
        lambda time, state: jnp.asarray([0.0]),
        1,
    )
    domain = phx.domain.Interval1d(-1.0, 1.0)
    objective = DeepSplittingRegressionObjective(
        problem,
        value_name="value",
        slice_index=1,
        labels=labels,
    )
    exact = domain.Parameter(jnp.asarray([0.5]))

    assert jnp.allclose(labels.next_values, 0.0)
    assert jnp.allclose(labels.source_values, 1.0)
    assert jnp.allclose(labels.source_controls, 0.0)
    assert jnp.allclose(labels.value_targets, 0.5)
    assert jnp.allclose(objective.loss({"value": exact}, batch=labels), 0.0)


def test_solve_deep_splitting_trains_distinct_slices_and_interpolates_field():
    paths = _paths()
    problem = _problem(paths)
    domain = phx.domain.Interval1d(-1.0, 1.0)
    solver = phx.solver.FunctionalSolver(
        functions={"value": domain.Parameter(jnp.asarray([0.0]))},
        constraints=(),
    )

    result = solve_deep_splitting(
        solver,
        problem,
        value_name="value",
        inner_num_iter=30,
        optim=optax.sgd(0.25),
        sampling_mode="fixed",
        fixed_paths=paths,
        validation_paths=paths,
        keep_best=False,
    )
    state = jnp.asarray([0.0])
    node_values = jnp.stack(
        tuple(result.solution.at_node(index, state) for index in range(3))
    )
    midpoint_value = jax.jit(lambda time: result.solution(time, state))(
        jnp.asarray(0.25)
    )

    assert jnp.allclose(node_values[:, 0], jnp.asarray([1.0, 0.5, 0.0]), atol=1e-8)
    assert jnp.allclose(midpoint_value, jnp.asarray([0.75]), atol=1e-8)
    assert jnp.allclose(result.solution.control(0.25, state), 0.0)
    assert result.completed_slices == 2
    assert result.diagnostics.passed
    assert jnp.all(result.diagnostics.one_step_rmse < 1e-8)
    assert result.solver.objectives == solver.objectives
    assert jnp.allclose(solver["value"].func(), jnp.asarray([0.0]))

    space_time = domain @ phx.domain.TimeInterval(0.0, 1.0)
    field = result.solution.as_domain_function(space_time)
    assert field.deps == ("t", "x")
    assert jnp.allclose(field.func(0.25, state), jnp.asarray([0.75]), atol=1e-8)
