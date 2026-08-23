import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _periodic_grid(size=8):
    axis = phx.discretization.UniformAxisSpec(
        size,
        endpoint=False,
        periodic=True,
    ).materialize(0.0, 1.0)
    grid = phx.discretization.PreparedTensorGrid((axis,))
    return phx.discretization.periodic_finite_difference(grid)


def test_spectral_noise_modes_are_weighted_orthonormal_and_scaled():
    discretization = _periodic_grid()
    basis = phx.stochastic.SpatialNoiseBasis.from_spectrum(
        discretization,
        lambda eigenvalues: jnp.exp(-0.2 * eigenvalues),
        rank=4,
    )
    modes = basis.modes.reshape((-1, basis.rank))
    weights = basis.quadrature_weights.reshape((-1, 1))
    gram = modes.T @ (weights * modes)

    assert basis.state_shape == discretization.state_shape
    assert basis.noise_shape == (4,)
    assert basis.diffusion.shape == (8, 4)
    assert basis.diffusion_matrix.shape == (8, 4)
    assert jnp.allclose(gram, jnp.eye(4), atol=1e-12)
    assert jnp.allclose(
        basis.diffusion,
        basis.modes * jnp.sqrt(basis.eigenvalues)[None, :],
    )


def test_mode_and_discrete_covariance_constructors_reconstruct_covariance():
    discretization = _periodic_grid(6)
    _, modes = discretization.eigenpairs(rank=3)
    from_modes = phx.stochastic.SpatialNoiseBasis.from_modes(
        modes,
        jnp.asarray([0.7, 0.3, 0.1]),
        quadrature_weights=discretization.quadrature_weights,
        mode_ids=("constant", "pair-a", "pair-b"),
        field_space_id=discretization.field_spaces[0].field_space_id,
    )
    factor = jnp.asarray(
        [
            [1.0, 0.0],
            [0.5, 0.2],
            [0.1, -0.4],
            [-0.3, 0.7],
            [0.2, 0.1],
            [0.0, -0.5],
        ]
    )
    covariance = factor @ factor.T
    from_covariance = phx.stochastic.SpatialNoiseBasis.from_discrete_covariance(
        covariance,
        state_shape=(6,),
        quadrature_weights=discretization.quadrature_weights,
        rank=6,
        field_space_id=discretization.field_spaces[0].field_space_id,
    )

    assert from_modes.mode_ids == ("constant", "pair-a", "pair-b")
    assert jnp.allclose(
        from_covariance.reconstructed_covariance(),
        covariance,
        atol=2e-12,
    )
    assert from_covariance.approximation is not None
    assert from_covariance.approximation.method == "dense_eigh"
    assert from_covariance.approximation.residual_estimate == 0.0


def test_kernel_covariance_uses_discretization_coordinates_and_weighted_kl():
    discretization = _periodic_grid(7)

    def kernel(left, right):
        return jnp.exp(-jnp.sum((left - right) ** 2, axis=-1) / 0.12)

    basis = phx.stochastic.SpatialNoiseBasis.from_kernel_covariance(
        kernel,
        discretization,
        rank=7,
    )
    points = discretization.points
    expected = kernel(points[:, None, :], points[None, :, :])

    assert jnp.allclose(basis.reconstructed_covariance(), expected, atol=2e-12)

    assert basis.approximation is not None
    assert basis.approximation.method == "pivoted_cholesky"
    assert basis.approximation.rank == 7
    assert basis.approximation.residual_kind == "relative_trace"
    assert basis.approximation.residual_estimate < 1e-10
    assert basis.approximation.converged


def test_randomized_covariance_operator_retains_weighted_kl_modes_and_seed():
    discretization = _periodic_grid(8)
    diagonal = jnp.arange(8.0, 0.0, -1.0)

    def covariance_operator(state):
        return diagonal * state

    basis = phx.stochastic.SpatialNoiseBasis.from_covariance_operator(
        covariance_operator,
        discretization,
        rank=3,
        key=jr.key(12),
        oversampling=5,
        tolerance=0.6,
        diagnostic_probes=4,
    )
    replay = phx.stochastic.SpatialNoiseBasis.from_covariance_operator(
        covariance_operator,
        discretization,
        rank=3,
        key=jr.key(12),
        oversampling=5,
        tolerance=0.6,
        diagnostic_probes=4,
    )
    approximation = basis.approximation

    assert approximation is not None
    assert approximation.method == "randomized_nystrom"
    assert approximation.seed == (0, 12)
    assert approximation.sketch_size == 8
    assert approximation.residual_kind == "relative_frobenius"
    assert approximation.converged
    assert basis.basis_id == replay.basis_id
    assert jnp.allclose(basis.eigenvalues, diagonal[:3] / 8.0, atol=2e-12)
    expected_relative_residual = jnp.linalg.vector_norm(diagonal[3:]) / (
        jnp.linalg.vector_norm(diagonal)
    )
    assert jnp.allclose(
        approximation.residual_estimate,
        expected_relative_residual,
        atol=2e-12,
    )


def test_basis_provenance_is_stable_and_changes_with_meaningful_inputs():
    grid = _periodic_grid(8)
    same_a = phx.stochastic.SpatialNoiseBasis.from_spectrum(grid, 0.2, rank=3)
    same_b = phx.stochastic.SpatialNoiseBasis.from_spectrum(grid, 0.2, rank=3)
    changed_spectrum = phx.stochastic.SpatialNoiseBasis.from_spectrum(grid, 0.3, rank=3)
    changed_rank = phx.stochastic.SpatialNoiseBasis.from_spectrum(grid, 0.2, rank=2)
    changed_grid = phx.stochastic.SpatialNoiseBasis.from_spectrum(
        _periodic_grid(9), 0.2, rank=3
    )

    assert same_a.basis_id == same_b.basis_id
    assert same_a.field_space_id == grid.field_spaces[0].field_space_id
    assert len(same_a.basis_id) == 64
    assert same_a.basis_id != changed_spectrum.basis_id
    assert same_a.basis_id != changed_rank.basis_id
    assert same_a.basis_id != changed_grid.basis_id


def test_noise_basis_provenance_reaches_wiener_realization():
    discretization = _periodic_grid(6)
    basis = phx.stochastic.SpatialNoiseBasis.from_spectrum(
        discretization,
        0.05,
        rank=2,
    )
    spde = phx.solver.semidiscretize_reaction_diffusion(
        jnp.zeros((6,)),
        discretization,
        t0=0.0,
        t1=0.1,
        kappa=0.02,
        noise_basis=basis,
    )
    realization = spde.wiener_realization(
        jr.key(4),
        tolerance=1e-4,
        label="basis-test",
    )

    assert realization.noise_shape == (2,)
    assert realization.noise_id == basis.basis_id
    assert realization.label == "basis-test"
    assert len(realization.realization_id) == 64
    assert jnp.array_equal(
        jr.key_data(realization.root_key),
        jr.key_data(jr.key(4)),
    )


def test_noise_basis_rejects_invalid_rank_modes_and_covariances():
    discretization = _periodic_grid(4)
    with pytest.raises(ValueError, match="rank must lie"):
        phx.stochastic.SpatialNoiseBasis.from_spectrum(
            discretization,
            1.0,
            rank=5,
        )
    with pytest.raises(ValueError, match="weighted Gram"):
        phx.stochastic.SpatialNoiseBasis.from_modes(
            jnp.ones((4, 2)),
            jnp.ones((2,)),
            quadrature_weights=discretization.quadrature_weights,
        )
    with pytest.raises(ValueError, match="non-negative"):
        phx.stochastic.SpatialNoiseBasis.from_modes(
            discretization.eigenpairs(rank=1)[1],
            jnp.asarray([-1.0]),
            quadrature_weights=discretization.quadrature_weights,
        )
    with pytest.raises(ValueError, match="symmetric"):
        phx.stochastic.SpatialNoiseBasis.from_discrete_covariance(
            jnp.asarray(
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            state_shape=(4,),
            quadrature_weights=discretization.quadrature_weights,
            rank=2,
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        phx.stochastic.SpatialNoiseBasis.from_discrete_covariance(
            jnp.diag(jnp.asarray([1.0, 1.0, 1.0, -0.1])),
            state_shape=(4,),
            quadrature_weights=discretization.quadrature_weights,
            rank=2,
        )


def test_semidiscrete_spde_preserves_declared_solution_concept_and_cutoff():
    discretization = _periodic_grid(6)
    basis = phx.stochastic.SpatialNoiseBasis.from_spectrum(
        discretization,
        0.05,
        rank=2,
    )
    declared = phx.stochastic.SPDESolutionSpec(
        "mild",
        noise_regularization="space_time_white",
        cutoff_id=basis.basis_id,
    )
    spde = phx.solver.semidiscretize_reaction_diffusion(
        jnp.zeros((6,)),
        discretization,
        t0=0.0,
        t1=0.1,
        kappa=0.02,
        noise_basis=basis,
        solution_spec=declared,
    )
    deterministic = phx.solver.semidiscretize_reaction_diffusion(
        jnp.zeros((6,)),
        discretization,
        t0=0.0,
        t1=0.1,
        kappa=0.02,
    )

    assert spde.solution_spec is declared
    assert spde.solution_spec.concept == "mild"
    assert spde.solution_spec.rough_forcing
    assert deterministic.solution_spec.noise_regularization == "none"

    with pytest.raises(ValueError, match="requires cutoff_id"):
        phx.solver.semidiscretize_reaction_diffusion(
            jnp.zeros((6,)),
            discretization,
            t0=0.0,
            t1=0.1,
            kappa=0.02,
            noise_basis=basis,
            solution_spec=phx.stochastic.SPDESolutionSpec(
                "mild",
                noise_regularization="space_time_white",
            ),
        )
