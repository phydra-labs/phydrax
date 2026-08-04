import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsp_linalg

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
    operator = lambda value: matrix @ value
    policy = phx.solver.MatrixFunctionPolicy("arnoldi", num_matvecs=2)
    expected_exponential = jsp_linalg.expm(step * matrix) @ vector
    expected_phi1 = jnp.linalg.solve(
        step * matrix,
        (jsp_linalg.expm(step * matrix) - jnp.eye(2)) @ vector,
    )

    exponential = phx.solver.matrix_exponential_action(
        operator,
        vector,
        step,
        policy=policy,
    )
    phi1 = phx.solver.matrix_phi1_action(
        operator,
        vector,
        step,
        policy=policy,
    )

    def approximate(value):
        return phx.solver.matrix_exponential_action(
            operator,
            value,
            step,
            policy=phx.solver.MatrixFunctionPolicy(
                "arnoldi",
                num_matvecs=2,
                differentiation="forward",
            ),
        )

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
    terminal = solution.states[:, -1]
    linear_eigenvalues = spde.semilinear_drift.compatible_noise_eigenvalues
    assert linear_eigenvalues is not None
    factors = jnp.where(
        jnp.abs(linear_eigenvalues) > 1e-12,
        jnp.expm1(2.0 * duration * linear_eigenvalues) / (2.0 * linear_eigenvalues),
        duration,
    )
    expected_covariance = jnp.einsum(
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
