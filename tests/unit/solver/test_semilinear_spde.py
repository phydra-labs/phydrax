import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsp_linalg
import opt_einsum as oe
import pytest

import phydrax as phx


def _periodic_discretization(size):
    axis = phx.domain.UniformAxisSpec(
        size,
        endpoint=False,
        periodic=True,
    ).materialize(0.0, 1.0)
    return phx.solver.TensorGridDiscretization((axis,))


def test_matrix_function_actions_match_dense_values_and_derivatives():
    matrix = jnp.asarray([[-1.5, 0.4], [-0.2, -0.7]])
    vector = jnp.asarray([0.8, -0.3])
    step = 0.2
    space = phx.linalg.ArraySpace(vector.shape, dtype=vector.dtype)
    operator = phx.linalg.FunctionLinearOperator(
        lambda value: matrix @ value,
        source=space,
        target=space,
    )
    policy = phx.linalg.MatrixFunctionPolicy("arnoldi", max_dimension=2)
    expected_exponential = jsp_linalg.expm(step * matrix) @ vector
    expected_phi1 = jnp.linalg.solve(
        step * matrix,
        (jsp_linalg.expm(step * matrix) - jnp.eye(2)) @ vector,
    )

    exponential = phx.linalg.matrix_exponential_action(
        operator,
        vector,
        step,
        policy=policy,
    ).value
    phi1 = phx.linalg.matrix_phi1_action(
        operator,
        vector,
        step,
        policy=policy,
    ).value

    def approximate(value):
        return phx.linalg.matrix_exponential_action(
            operator,
            value,
            step,
            policy=policy,
        ).value

    tangent = jnp.asarray([-0.1, 0.5])
    _, approximate_jvp = jax.jvp(approximate, (vector,), (tangent,))
    expected_jvp = jsp_linalg.expm(step * matrix) @ tangent
    approximate_gradient = jax.grad(lambda value: jnp.sum(approximate(value) ** 2))(
        vector
    )
    expected_gradient = jax.grad(
        lambda value: jnp.sum((jsp_linalg.expm(step * matrix) @ value) ** 2)
    )(vector)

    assert jnp.allclose(exponential, expected_exponential, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(phi1, expected_phi1, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(approximate_jvp, expected_jvp, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(
        approximate_gradient,
        expected_gradient,
        rtol=1e-10,
        atol=1e-10,
    )


def test_semilinear_solver_propagates_linear_heat_mode_exactly():
    discretization = _periodic_discretization(8)
    duration = 0.3
    diffusivity = 0.04
    initial = jnp.sin(2.0 * jnp.pi * discretization.axes[0].nodes)
    spde = phx.solver.semidiscretize_reaction_diffusion(
        initial,
        discretization,
        t0=0.0,
        t1=duration,
        kappa=diffusivity,
    )
    solution = phx.solver.solve_semilinear_spde(
        spde,
        save_times=jnp.asarray([0.0, duration]),
        dt=duration,
    )
    expected = (
        jsp_linalg.expm(duration * diffusivity * discretization.laplacian_matrix())
        @ initial
    )

    assert solution.solver_name == "SemilinearExponentialEuler"
    assert solution.solver_id == "solver:semilinear:exponential_euler"
    assert solution.resolved_method.startswith("exponential_euler:")
    assert solution.stats["exact_stochastic_convolution"] is False
    assert jnp.array_equal(solution.states[0], initial)
    assert jnp.allclose(solution.states[-1], expected, rtol=1e-10, atol=1e-10)


def test_exact_modal_stochastic_convolution_replays_and_matches_covariance():
    discretization = _periodic_discretization(4)
    duration = 0.08
    diffusivity = 0.1
    basis = phx.solver.SpatialNoiseBasis.from_spectrum(
        discretization,
        0.03,
        rank=2,
    )
    spde = phx.solver.semidiscretize_reaction_diffusion(
        jnp.zeros(discretization.state_shape),
        discretization,
        t0=0.0,
        t1=duration,
        kappa=diffusivity,
        noise_basis=basis,
    )
    realization = spde.wiener_realization(
        jr.key(11),
        sample_shape=(2048,),
        label="exact-convolution",
    )

    def solve(selected_realization):
        return phx.solver.solve_semilinear_spde(
            spde,
            save_times=jnp.asarray([duration]),
            realization=selected_realization,
            dt=duration,
        )

    solution = solve(realization)
    replay = solve(realization)
    changed = solve(
        spde.wiener_realization(
            jr.key(12),
            sample_shape=(2048,),
            label="changed-convolution",
        )
    )
    assert spde.semilinear_drift is not None
    terminal = solution.states[:, -1]
    linear_eigenvalues = spde.semilinear_drift.compatible_noise_eigenvalues
    assert linear_eigenvalues is not None
    factors = jnp.where(
        jnp.abs(linear_eigenvalues) > 1e-12,
        jnp.expm1(2.0 * duration * linear_eigenvalues) / (2.0 * linear_eigenvalues),
        duration,
    )
    expected_covariance = oe.contract(
        "ir,r,jr->ij",
        basis.modes.reshape((-1, basis.rank)),
        basis.eigenvalues * factors,
        basis.modes.reshape((-1, basis.rank)),
    )
    empirical_covariance = jnp.cov(terminal, rowvar=False)
    relative_covariance_error = jnp.linalg.norm(
        empirical_covariance - expected_covariance
    ) / jnp.linalg.norm(expected_covariance)
    trajectory = solution.to_stochastic_trajectory(
        initial_state=jnp.zeros(discretization.state_shape),
        initial_time=0.0,
        realization_axes=("path",),
        state_axes=("space",),
        discretization_id=discretization.discretization_id,
        basis_id=basis.basis_id,
    )

    assert solution.stats["exact_stochastic_convolution"] is True
    assert jnp.array_equal(solution.states, replay.states)
    assert not jnp.array_equal(solution.states, changed.states)
    assert relative_covariance_error < 0.1
    assert trajectory.states.shape == (2048, 2, 4)
    assert trajectory.realizations == (realization,)
    assert trajectory.discretization_id == discretization.discretization_id
    assert trajectory.basis_id == basis.basis_id


def _geometric_spde(*, duration, rate=-0.2, noise=0.7, structure="commutative"):
    discretization = _periodic_discretization(2)
    initial = jnp.asarray([0.8, 1.3])
    operator = phx.linalg.DenseLinearOperator(
        rate * jnp.eye(2),
        operator_id="geometric-linear-drift",
    )
    spectral = phx.linalg.SpectralMatrixRepresentation(
        operator,
        jnp.full((2,), rate),
        jnp.eye(2),
        jnp.eye(2),
        representation_id="geometric-linear-drift",
    )
    spde = phx.solver.semidiscretize_semilinear_spde(
        operator,
        None,
        initial,
        discretization,
        t0=0.0,
        t1=duration,
        operator_id="geometric-linear-drift",
        diffusion=lambda time, state, args: noise * jnp.diag(state),
        noise_shape=(2,),
        basis_id="geometric-commutative-noise",
        noise_structure=structure,
        spectral_representation=spectral,
    )
    return spde, initial


def test_multiplicative_exponential_euler_uses_global_wiener_increments():
    duration, rate, noise = 0.2, -0.2, 0.7
    spde, initial = _geometric_spde(
        duration=duration,
        rate=rate,
        noise=noise,
    )
    realization = spde.wiener_realization(
        jr.key(31),
        sample_shape=(32,),
        tolerance=1e-5,
    )
    solution = phx.solver.solve_semilinear_spde(
        spde,
        save_times=jnp.asarray([duration]),
        realization=realization,
        dt=duration,
        scheme="exponential_euler",
        fallback="error",
    )
    increments = realization.increments(
        jnp.asarray([0.0]),
        jnp.asarray([duration]),
    )[:, 0]
    expected = jnp.exp(rate * duration) * (initial + noise * initial * increments)

    assert solution.solver_name == "SemilinearExponentialEuler"
    assert solution.stats["scheme"] == "exponential_euler"
    assert solution.stats["uses_realization_increments"]
    assert jnp.allclose(solution.states[:, 0], expected, rtol=1e-11, atol=1e-11)


def test_exponential_milstein_matches_one_step_factor_jvp_and_is_higher_order():
    duration, rate, noise = 0.5, -0.2, 0.7
    spde, initial = _geometric_spde(
        duration=duration,
        rate=rate,
        noise=noise,
    )
    realization = spde.wiener_realization(
        jr.key(32),
        sample_shape=(512,),
        tolerance=1e-5,
    )
    one_step = phx.solver.solve_semilinear_spde(
        spde,
        save_times=jnp.asarray([duration]),
        realization=realization,
        dt=duration,
        scheme="exponential_milstein",
        fallback="error",
    )
    total_increment = realization.increments(
        jnp.asarray([0.0]),
        jnp.asarray([duration]),
    )[:, 0]
    expected_one_step = (
        jnp.exp(rate * duration)
        * initial
        * (
            1.0
            + noise * total_increment
            + 0.5 * noise**2 * (total_increment**2 - duration)
        )
    )
    exact = initial * jnp.exp(
        (rate - 0.5 * noise**2) * duration + noise * total_increment
    )

    def terminal(scheme, step):
        return phx.solver.solve_semilinear_spde(
            spde,
            save_times=jnp.asarray([duration]),
            realization=realization,
            dt=step,
            scheme=scheme,
            fallback="error",
        ).states[:, 0]

    euler_fine = terminal("exponential_euler", duration / 8.0)
    milstein_coarse = terminal("exponential_milstein", duration / 4.0)
    milstein_fine = terminal("exponential_milstein", duration / 8.0)
    euler_error = jnp.sqrt(jnp.mean((euler_fine - exact) ** 2))
    milstein_coarse_error = jnp.sqrt(jnp.mean((milstein_coarse - exact) ** 2))
    milstein_fine_error = jnp.sqrt(jnp.mean((milstein_fine - exact) ** 2))

    assert one_step.solver_name == "SemilinearExponentialMilstein"
    assert jnp.allclose(
        one_step.states[:, 0],
        expected_one_step,
        rtol=1e-11,
        atol=1e-11,
    )
    assert milstein_fine_error < 0.7 * milstein_coarse_error
    assert milstein_fine_error < 0.4 * euler_error


def test_exponential_milstein_rejects_undeclared_commutativity():
    spde, _ = _geometric_spde(duration=0.1, structure="general")
    realization = spde.wiener_realization(jr.key(33), tolerance=1e-5)

    with pytest.raises(ValueError, match="declared commutative noise"):
        phx.solver.solve_semilinear_spde(
            spde,
            save_times=jnp.asarray([0.1]),
            realization=realization,
            dt=0.1,
            scheme="exponential_milstein",
            fallback="error",
        )
