import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _periodic_discretization(size):
    axis = phx.discretization.FourierAxisSpec(size).materialize(0.0, 1.0)
    return phx.discretization.TensorSpectralDiscretization.from_axes((axis,))


def test_stochastic_heat_ensemble_matches_semidiscrete_gaussian_moments():
    discretization = _periodic_discretization(4)
    kappa, duration = 0.1, 0.05
    initial = discretization.project(jnp.asarray([0.7, -0.2, 0.1, 0.4]))
    basis = phx.stochastic.SpatialNoiseBasis.from_spectrum(
        discretization,
        0.04,
        rank=2,
    )
    spde = phx.solver.semidiscretize_reaction_diffusion(
        initial,
        discretization,
        t0=0.0,
        t1=duration,
        kappa=kappa,
        noise_basis=basis,
    )
    realization = spde.wiener_realization(
        jr.key(20),
        sample_shape=(2048,),
        tolerance=1e-4,
        label="heat-moments",
    )
    solution = phx.solver.solve_diffrax_ensemble(
        spde.problem,
        save_times=jnp.asarray([duration]),
        realization=realization,
        dt0=1e-3,
    )
    terminal = solution.states[:, 0, :]

    laplacian_values = discretization.laplacian_eigenvalues().reshape((-1,))
    expected_mean = jnp.exp(-duration * kappa * laplacian_values) * initial
    retained_indices = jnp.argsort(laplacian_values)[: basis.rank]
    laplacian_eigenvalues = laplacian_values[retained_indices]
    expected_covariance = jnp.zeros((4, 4), dtype=terminal.dtype)
    for mode in range(basis.rank):
        rate = float(kappa * laplacian_eigenvalues[mode])
        factor = (
            duration
            if rate == 0.0
            else (1.0 - jnp.exp(-2.0 * rate * duration)) / (2.0 * rate)
        )
        column = basis.diffusion_matrix[:, mode]
        expected_covariance = expected_covariance + factor * jnp.outer(
            column, jnp.conj(column)
        )
    empirical_mean = jnp.mean(terminal, axis=0)
    centered = terminal - empirical_mean
    empirical_covariance = jnp.conj(centered).T @ centered / float(terminal.shape[0] - 1)
    relative_covariance_error = jnp.linalg.norm(
        empirical_covariance - expected_covariance
    ) / jnp.linalg.norm(expected_covariance)

    assert solution.states.shape == (2048, 1, 4)
    assert jnp.all(solution.successful)
    assert jnp.allclose(empirical_mean, expected_mean, atol=6e-3, rtol=2e-2)
    assert relative_covariance_error < 0.12
    assert solution.realization is realization
    assert solution.realization.noise_id == basis.basis_id

    predictive = solution.to_predictive(
        sample_dim="path",
        time_dim="time",
        state_dims=("mode",),
    )
    assert predictive.sample_axes == (phx.uq.SampleAxis("path", "process"),)
    assert predictive.samples.shape == (2048, 1, 4)


def test_semidiscrete_heat_replays_realization_and_changes_with_key():
    discretization = _periodic_discretization(5)
    basis = phx.stochastic.SpatialNoiseBasis.from_spectrum(
        discretization,
        0.02,
        rank=2,
    )
    initial = discretization.project(jnp.sin(2.0 * jnp.pi * discretization.axes[0].nodes))
    spde = phx.solver.semidiscretize_reaction_diffusion(
        initial,
        discretization,
        t0=0.0,
        t1=0.03,
        kappa=0.04,
        noise_basis=basis,
    )
    realization = spde.wiener_realization(
        jr.key(21),
        sample_shape=(64,),
        tolerance=1e-4,
    )

    def solve(selected_realization):
        return phx.solver.solve_diffrax_ensemble(
            spde.problem,
            save_times=jnp.asarray([0.03]),
            realization=selected_realization,
            dt0=1e-3,
        )

    first = solve(realization)
    replay = solve(realization)
    changed = solve(
        spde.wiener_realization(
            jr.key(22),
            sample_shape=(64,),
            tolerance=1e-4,
        )
    )

    assert jnp.array_equal(first.states, replay.states)
    assert not jnp.array_equal(first.states, changed.states)


def test_stochastic_allen_cahn_semidiscretization_is_finite_and_reproducible():
    discretization = _periodic_discretization(6)
    basis = phx.stochastic.SpatialNoiseBasis.from_spectrum(
        discretization,
        0.01,
        rank=3,
    )
    method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(3)
    ).prepare(
        discretization,
        required_polynomial_degree=3,
        nonlinear=True,
    )
    initial = discretization.project(
        0.25 * jnp.cos(2.0 * jnp.pi * discretization.axes[0].nodes)
    )
    spde = phx.solver.semidiscretize_reaction_diffusion(
        initial,
        discretization,
        t0=0.0,
        t1=0.04,
        kappa=0.02,
        reaction=lambda t, state, args: method.nonlinear_action(
            state,
            lambda values: values - values**3,
        ),
        noise_basis=basis,
    )
    realization = spde.wiener_realization(
        jr.key(23),
        sample_shape=(32,),
        tolerance=1e-4,
    )
    first = phx.solver.solve_diffrax_ensemble(
        spde.problem,
        save_times=jnp.asarray([0.02, 0.04]),
        realization=realization,
        dt0=1e-3,
    )
    replay = phx.solver.solve_diffrax_ensemble(
        spde.problem,
        save_times=jnp.asarray([0.02, 0.04]),
        realization=realization,
        dt0=1e-3,
    )
    physical_states = jax.vmap(
        jax.vmap(
            lambda coefficients: discretization.reconstruct(
                coefficients,
                real_output=False,
            )
        )
    )(first.states)

    assert first.states.shape == (32, 2, 6)
    assert jnp.all(jnp.isfinite(first.states))
    assert jnp.array_equal(first.states, replay.states)
    assert jnp.max(jnp.abs(jnp.imag(physical_states))) < 1e-10
    assert first.temporal_evidence is not None
    assert first.temporal_evidence.state_packing is not None
    assert first.temporal_evidence.state_packing.strategy == "real_imag"


def test_two_dimensional_tensor_state_preserves_channels_and_noise_axes():
    x_axis = phx.discretization.FourierAxisSpec(4).materialize(0.0, 1.0)
    y_axis = phx.discretization.FourierAxisSpec(5).materialize(0.0, 1.0)
    discretization = phx.discretization.TensorSpectralDiscretization.from_axes(
        (x_axis, y_axis)
    )
    x, y = x_axis.nodes[:, None], y_axis.nodes[None, :]
    scalar = 0.1 * jnp.sin(2.0 * jnp.pi * x) * jnp.cos(2.0 * jnp.pi * y)
    initial = discretization.project(jnp.stack((scalar, -scalar), axis=-1))
    state_shape = initial.shape
    weights = jnp.ones(state_shape)
    mode = jnp.zeros(state_shape).at[0, 0, :].set(1.0 / jnp.sqrt(2.0))
    basis = phx.stochastic.SpatialNoiseBasis.from_modes(
        mode[..., None],
        jnp.asarray([0.005]),
        quadrature_weights=weights,
        state_shape=state_shape,
        mode_ids=("shared-channel-mode",),
        field_space_id=discretization.modal_space.field_space_id,
    )
    spde = phx.solver.semidiscretize_reaction_diffusion(
        initial,
        discretization,
        t0=0.0,
        t1=0.01,
        kappa=0.01,
        noise_basis=basis,
    )
    solution = phx.solver.solve_diffrax_ensemble(
        spde.problem,
        save_times=jnp.asarray([0.01]),
        realization=spde.wiener_realization(
            jr.key(24),
            sample_shape=(4,),
            tolerance=1e-4,
        ),
        dt0=1e-3,
    )

    assert spde.state_shape == (4, 5, 2)
    assert spde.noise_shape == (1,)
    coefficient = jnp.asarray(
        spde.problem.wiener_terms[0].coefficient(jnp.asarray(0.0), initial, None)
    )
    assert coefficient.shape == (
        4,
        5,
        2,
        1,
    )
    assert solution.states.shape == (4, 1, 4, 5, 2)
    assert jnp.all(jnp.isfinite(solution.states))


def test_refined_grid_changes_discretization_and_basis_provenance():
    coarse = _periodic_discretization(6)
    refined = _periodic_discretization(10)
    coarse_basis = phx.stochastic.SpatialNoiseBasis.from_spectrum(coarse, 0.02, rank=2)
    refined_basis = phx.stochastic.SpatialNoiseBasis.from_spectrum(refined, 0.02, rank=2)

    assert coarse.discretization_id != refined.discretization_id
    assert coarse_basis.basis_id != refined_basis.basis_id
    assert coarse_basis.field_space_id == coarse.field_spaces[0].field_space_id
    assert refined_basis.field_space_id == refined.field_spaces[0].field_space_id


def test_semidiscrete_stratonovich_geometric_noise_matches_analytic_moments():
    discretization = _periodic_discretization(2)
    initial = jnp.asarray([1.0, 1.5])
    rate, noise, duration = 0.2, 0.4, 0.2
    spde = phx.solver.semidiscretize_spde(
        lambda t, state, args: rate * state,
        initial,
        discretization,
        t0=0.0,
        t1=duration,
        diffusion=lambda t, state, args: noise * jnp.diag(state),
        noise_shape=(2,),
        basis_id="geometric-state-noise",
        interpretation="stratonovich",
    )
    solution = phx.solver.solve_diffrax_ensemble(
        spde.problem,
        save_times=jnp.asarray([duration]),
        realization=spde.wiener_realization(
            jr.key(25),
            sample_shape=(2048,),
            tolerance=1e-4,
        ),
        dt0=2e-3,
    )
    terminal = solution.states[:, 0, :]
    expected_mean = initial * jnp.exp((rate + 0.5 * noise**2) * duration)
    expected_variance = (
        initial**2
        * jnp.exp((2.0 * rate + noise**2) * duration)
        * (jnp.exp(noise**2 * duration) - 1.0)
    )

    assert spde.problem.interpretation == "stratonovich"
    assert solution.interpretation == "stratonovich"
    assert solution.solver_name == "EulerHeun"
    assert jnp.allclose(jnp.mean(terminal, axis=0), expected_mean, rtol=0.03)
    assert jnp.allclose(jnp.var(terminal, axis=0), expected_variance, rtol=0.15)
