import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _additive_problem(*, rate=0.0, initial=0.0, interpretation="ito"):
    return phx.solver.DifferentialProblem(
        lambda time, state, value: value * state,
        jnp.asarray([initial]),
        t0=0.0,
        t1=1.0,
        args=rate,
        wiener_terms=(
            phx.solver.WienerTerm(
                "noise",
                lambda time, state, args: jnp.ones(state.shape + (1,)),
                (1,),
                structure="additive",
            ),
        ),
        interpretation=interpretation,
    )


def _plan(intervals=4, **kwargs):
    return phx.solver.MarkovCubaturePlan(
        phx.discretization.TemporalMesh.uniform(
            0.0,
            1.0,
            intervals,
            role="driver",
        ),
        phx.integration.GaussianCubatureRule(1, 3),
        **kwargs,
    )


def _moments(solution, index=-1):
    mask = solution.mask[index]
    weights = jnp.where(mask, jnp.exp(solution.log_weights[index]), 0.0)
    values = solution.points[index, :, 0]
    return jnp.sum(weights * values), jnp.sum(weights * values**2)


def test_weak_euler_recombination_preserves_linear_gaussian_moments():
    intervals = 4
    rate = 0.2
    initial = 1.0
    solution = phx.solver.solve_markov_cubature(
        _additive_problem(rate=rate, initial=initial),
        _plan(intervals),
    )
    mean, second = _moments(solution)
    step = 1.0 / intervals
    multiplier = 1.0 + rate * step
    expected_mean = initial * multiplier**intervals
    expected_variance = step * sum(
        multiplier ** (2 * power) for power in range(intervals)
    )

    assert solution.successful
    assert jnp.all(solution.valid)
    assert jnp.all(solution.diagnostics.statuses == 0)
    assert jnp.all(solution.diagnostics.retained_points <= 3)
    assert jnp.all(solution.diagnostics.expanded_points <= 6)
    assert jnp.all(solution.log_weights[solution.mask] <= 0.0)
    assert jnp.allclose(mean, expected_mean, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(
        second,
        expected_mean**2 + expected_variance,
        rtol=1e-10,
        atol=1e-10,
    )

    estimate = phx.integration.integrate(
        lambda states: states[..., 0] ** 2,
        solution.measure(),
    )
    assert estimate.successful
    assert estimate.error_estimate is None
    assert jnp.allclose(jnp.asarray(estimate.value.data), second, atol=1e-12)


def test_weak_solver_is_jittable_and_uses_frozen_support_weight_derivatives():
    plan = _plan(4)

    def terminal_mean(rate):
        solution = phx.solver.solve_markov_cubature(
            _additive_problem(rate=rate, initial=1.0),
            plan,
        )
        return _moments(solution)[0]

    rate = jnp.asarray(0.2)
    compiled = jax.jit(terminal_mean)(rate)
    gradient = jax.grad(terminal_mean)(rate)

    assert jnp.allclose(compiled, 1.05**4, atol=1e-12)
    assert jnp.isfinite(gradient)
    assert jnp.allclose(gradient, 1.05**3, atol=1e-11)


def test_markov_solver_preserves_float32_state_dtype_under_jit():
    problem = phx.solver.DifferentialProblem(
        lambda time, state, args: -jnp.asarray(0.2, dtype=state.dtype) * state,
        jnp.asarray([1.0], dtype=jnp.float32),
        t0=0.0,
        t1=1.0,
        wiener_terms=(
            phx.solver.WienerTerm(
                "noise",
                lambda time, state, args: jnp.ones(state.shape + (1,), dtype=state.dtype),
                (1,),
                structure="additive",
            ),
        ),
    )

    solution = jax.jit(lambda: phx.solver.solve_markov_cubature(problem, _plan(2)))()

    assert solution.points.dtype == jnp.float32
    assert solution.successful


def test_temporal_masks_hold_the_law_without_recording_fake_expansions():
    mesh = phx.discretization.TemporalMesh(
        jnp.asarray([0.0, 0.25, 0.5, 1.0]),
        role="driver",
        active_intervals=jnp.asarray([True, False, True]),
    )
    plan = phx.solver.MarkovCubaturePlan(
        mesh,
        phx.integration.GaussianCubatureRule(1, 3),
    )
    problem = phx.solver.DifferentialProblem(
        lambda time, state, args: jnp.ones_like(state),
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
        wiener_terms=(
            phx.solver.WienerTerm(
                "zero-noise",
                lambda time, state, args: jnp.zeros(state.shape + (1,)),
                (1,),
                structure="additive",
            ),
        ),
    )

    solution = jax.jit(lambda: phx.solver.solve_markov_cubature(problem, plan))()
    means = jnp.stack([_moments(solution, index)[0] for index in range(4)])

    assert jnp.allclose(means, jnp.asarray([0.0, 0.25, 0.25, 0.75]))
    assert jnp.array_equal(
        solution.diagnostics.expanded_points,
        jnp.asarray([2, 0, 2]),
    )
    assert jnp.all(solution.valid)


def test_dynamic_nonfinite_failure_returns_a_status_when_throw_is_disabled():
    problem = phx.solver.DifferentialProblem(
        lambda time, state, args: jnp.full_like(state, jnp.nan),
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
        wiener_terms=(
            phx.solver.WienerTerm(
                "noise",
                lambda time, state, args: jnp.ones(state.shape + (1,)),
                (1,),
                structure="additive",
            ),
        ),
    )

    solution = phx.solver.solve_markov_cubature(problem, _plan(2, throw=False))

    assert solution.status == int(phx.solver.MarkovCubatureStatus.NONFINITE_DYNAMICS)
    assert not solution.successful
    assert not solution.valid[-1]


def test_markov_cubature_rejects_unsupported_static_problem_contracts():
    deterministic = phx.solver.DifferentialProblem(
        lambda time, state, args: state,
        jnp.asarray([1.0]),
        t0=0.0,
        t1=1.0,
    )
    with pytest.raises(ValueError, match="requires a stochastic"):
        phx.solver.solve_markov_cubature(deterministic, _plan())

    multiplicative_stratonovich = phx.solver.DifferentialProblem(
        lambda time, state, args: jnp.zeros_like(state),
        jnp.asarray([1.0]),
        t0=0.0,
        t1=1.0,
        wiener_terms=(
            phx.solver.WienerTerm(
                "noise",
                lambda time, state, args: state[..., None],
                (1,),
                structure="commutative",
            ),
        ),
        interpretation="stratonovich",
    )
    with pytest.raises(NotImplementedError, match="only for additive noise"):
        phx.solver.solve_markov_cubature(multiplicative_stratonovich, _plan())

    with pytest.raises(ValueError, match="exceeding maximum_expanded_particles"):
        phx.solver.solve_markov_cubature(
            _additive_problem(),
            _plan(maximum_expanded_particles=5),
        )
