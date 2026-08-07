import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _linear_problem(rate):
    return phx.solver.RoughDifferentialProblem(
        lambda time, state, args: rate * state[..., None],
        jnp.asarray([1.2]),
        driver_dimension=1,
        problem_id="linear-geometric-rde",
    )


def test_piecewise_linear_lift_and_coarsening_satisfy_chens_identity():
    times = jnp.linspace(0.0, 1.0, 5)
    values = jnp.asarray([[0.0, 0.0], [0.3, -0.2], [0.1, 0.4], [0.8, 0.5], [0.7, 1.0]])
    rough_path = phx.stochastic.GeometricRoughPath.from_values(times, values)
    coarse = rough_path.coarsen((0, 2, 4))
    fine_terminal = rough_path.terminal_signature
    coarse_terminal = coarse.terminal_signature

    assert jnp.allclose(
        coarse.first_level[0], jnp.sum(rough_path.first_level[:2], axis=0)
    )
    assert jnp.allclose(
        coarse.first_level[1], jnp.sum(rough_path.first_level[2:], axis=0)
    )
    assert jnp.allclose(fine_terminal[0], coarse_terminal[0])
    assert jnp.allclose(fine_terminal[1], coarse_terminal[1])
    symmetric = 0.5 * (coarse.second_level + jnp.swapaxes(coarse.second_level, -1, -2))
    expected = 0.5 * jnp.einsum(
        "...i,...j->...ij", coarse.first_level, coarse.first_level
    )
    assert jnp.allclose(symmetric, expected)


def test_davie_step_uses_second_level_factor_jvp_and_improves_smooth_path_error():
    rate = 0.8
    one_step_path = phx.stochastic.GeometricRoughPath.from_values(
        jnp.asarray([0.0, 0.3]),
        jnp.asarray([[0.0], [0.3]]),
    )
    one_step = phx.solver.solve_rough_differential(
        _linear_problem(rate),
        one_step_path,
        save_times=jnp.asarray([0.3]),
        solver=phx.solver.Davie(),
    )
    expected_one_step = 1.2 * (1.0 + rate * 0.3 + 0.5 * rate**2 * 0.3**2)

    times = jnp.linspace(0.0, 1.0, 17)
    smooth_path = phx.stochastic.GeometricRoughPath.from_values(times, times[:, None])
    euler = phx.solver.solve_rough_differential(
        _linear_problem(rate),
        smooth_path,
        save_times=jnp.asarray([1.0]),
        solver=phx.solver.RoughEuler(),
    )
    davie = phx.solver.solve_rough_differential(
        _linear_problem(rate),
        smooth_path,
        save_times=jnp.asarray([1.0]),
        solver=phx.solver.Davie(),
    )
    exact = 1.2 * jnp.exp(rate)
    euler_error = jnp.abs(euler.states[0, 0] - exact)
    davie_error = jnp.abs(davie.states[0, 0] - exact)

    assert jnp.allclose(one_step.states[0, 0], expected_one_step, atol=1e-12)
    assert davie_error < 0.08 * euler_error


def test_fractional_gaussian_rough_dynamics_track_linear_geometric_solution():
    process = phx.stochastic.FractionalGaussianProcess(
        0.7,
        0.4,
        process_id="fractional-rough-driver",
    )
    realization = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(70),
        jnp.linspace(0.0, 1.0, 65),
        sample_shape=(32,),
    )
    rough_path = phx.stochastic.GeometricRoughPath.from_fractional_gaussian(realization)
    rate = 0.6
    solution = phx.solver.solve_rough_differential(
        _linear_problem(rate),
        rough_path,
        save_times=jnp.asarray([1.0]),
    )
    terminal_driver = realization.values[:, -1, 0]
    exact = 1.2 * jnp.exp(rate * terminal_driver)
    relative_rmse = jnp.sqrt(
        jnp.mean((solution.states[:, 0, 0] - exact) ** 2)
    ) / jnp.sqrt(jnp.mean(exact**2))
    trajectory = solution.to_stochastic_trajectory(
        realization_axes=("path",),
        state_axes=("state",),
    )

    assert relative_rmse < 2e-3
    assert jnp.all(solution.successful)
    assert isinstance(solution.solver, phx.solver.Davie)
    assert solution.control is rough_path
    assert trajectory.realizations == (realization,)


def test_step_two_fractional_solver_rejects_hurst_requiring_level_three():
    process = phx.stochastic.FractionalGaussianProcess(0.3, 0.2)
    realization = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(71),
        jnp.linspace(0.0, 1.0, 9),
    )
    rough_path = phx.stochastic.GeometricRoughPath.from_fractional_gaussian(realization)

    with pytest.raises(ValueError, match="Hurst > 1/3"):
        phx.solver.solve_rough_differential(
            _linear_problem(0.5),
            rough_path,
        )
