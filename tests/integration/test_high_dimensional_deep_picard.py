import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax

import phydrax as phx
from phydrax.solver._deep_picard import solve_deep_picard
from phydrax.stochastic._bsde import BSDEPathBatch, BSDEProblem
from phydrax.stochastic._feynman_kac import FeynmanKacSamplingPlan


class _LinearHJBValue(eqx.Module):
    time_coefficient: jnp.ndarray

    def __call__(self, time, state, *, key=None):
        del key
        return jnp.asarray([jnp.mean(state) + self.time_coefficient * (1.0 - time)])


def test_dimension_100_quadratic_hjb_deep_picard_smoke():
    dimension = 100
    times = jnp.asarray([0.0, 1.0])
    placeholder = BSDEPathBatch(
        times,
        jnp.zeros((1, 2, dimension)),
        jnp.zeros((1, 1, dimension)),
        sample_shape=(1,),
        state_shape=(dimension,),
        noise_shape=(dimension,),
        path_id="placeholder",
        process_id="hjb-100",
    )
    problem = BSDEProblem(
        lambda key: placeholder,
        lambda time, state, args: jnp.zeros_like(state),
        lambda time, state, args: jnp.eye(dimension),
        lambda time, state, value, control, args: jnp.asarray(
            [0.5 * jnp.sum(control**2)]
        ),
        lambda state, args: jnp.asarray([jnp.mean(state)]),
        state_shape=(dimension,),
        noise_shape=(dimension,),
        output_shape=(1,),
        problem_id="quadratic-hjb-100",
        process_id="hjb-100",
    )
    domain = phx.domain.HyperRectangle(
        jnp.full((dimension,), -4.0),
        jnp.full((dimension,), 4.0),
        label="x",
    ) @ phx.domain.TimeInterval(0.0, 1.0)
    value = domain.Function("t", "x")(_LinearHJBValue(jnp.asarray(0.0)))
    solver = phx.solver.FunctionalSolver(
        functions={"value": value},
        terms=(),
    )
    plan = FeynmanKacSamplingPlan(
        terminal_time=1.0,
        sampling_mode="queries",
        num_paths_per_query=32,
        num_time_steps=4,
        antithetic=True,
        refresh_mode="fixed",
    )
    query_times = jnp.linspace(0.0, 0.9, 16)
    query_states = jr.normal(jr.key(10), (16, dimension))

    result = solve_deep_picard(
        solver,
        problem,
        value_name="value",
        sampling_plan=plan,
        num_picard_steps=1,
        inner_num_iter=120,
        optim=optax.adam(0.01),
        query_times=query_times,
        query_states=query_states,
        initial_source="current",
        seed=11,
        jit=True,
        keep_best=False,
    )

    coefficient = float(result.solver["value"].func.function.time_coefficient)
    expected = 0.5 / dimension
    assert abs(coefficient - expected) < 5e-4
    assert result.diagnostics.target_rmse[-1] < 5e-4
    assert result.diagnostics.finite[-1]
