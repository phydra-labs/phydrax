import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
import phydrax._spectral as spectral


def test_periodic_finite_difference_laplacian_matches_discrete_mode():
    axis = phx.domain.UniformAxisSpec(
        8,
        endpoint=False,
        periodic=True,
    ).materialize(0.0, 1.0)
    discretization = phx.solver.TensorGridDiscretization((axis,))
    state = jnp.sin(2.0 * jnp.pi * axis.nodes)
    spacing = axis.nodes[1] - axis.nodes[0]
    eigenvalue = 2.0 * (jnp.cos(2.0 * jnp.pi / 8.0) - 1.0) / spacing**2

    actual = discretization.laplacian(state)

    assert discretization.state_shape == (8,)
    assert discretization.boundary_conditions == ("periodic",)
    assert jnp.allclose(actual, eigenvalue * state, atol=1e-12)


def test_fourier_sine_and_cosine_laplacians_respect_boundary_semantics():
    fourier_axis = phx.domain.FourierAxisSpec(16).materialize(0.0, 1.0)
    sine_axis = phx.domain.SineAxisSpec(15).materialize(0.0, 1.0)
    cosine_axis = phx.domain.CosineAxisSpec(16).materialize(0.0, 1.0)
    fourier = phx.solver.TensorGridDiscretization((fourier_axis,))
    sine = phx.solver.TensorGridDiscretization((sine_axis,))
    cosine = phx.solver.TensorGridDiscretization((cosine_axis,))
    fourier_state = jnp.sin(4.0 * jnp.pi * fourier_axis.nodes)
    sine_state = jnp.sin(jnp.pi * sine_axis.nodes)
    cosine_state = jnp.cos(jnp.pi * cosine_axis.nodes)

    assert fourier.boundary_conditions == ("periodic",)
    assert sine.boundary_conditions == ("homogeneous_dirichlet",)
    assert cosine.boundary_conditions == ("homogeneous_neumann",)
    assert jnp.allclose(
        fourier.laplacian(fourier_state),
        -((4.0 * jnp.pi) ** 2) * fourier_state,
        atol=2e-11,
    )
    assert jnp.allclose(
        sine.laplacian(sine_state),
        -(jnp.pi**2) * sine_state,
        atol=2e-11,
    )
    assert jnp.allclose(
        cosine.laplacian(cosine_state),
        -(jnp.pi**2) * cosine_state,
        atol=2e-11,
    )


def test_tensor_grid_laplacian_preserves_channels_and_compiles():
    x_axis = phx.domain.FourierAxisSpec(8).materialize(0.0, 1.0)
    y_axis = phx.domain.FourierAxisSpec(10).materialize(0.0, 1.0)
    discretization = phx.solver.TensorGridDiscretization((x_axis, y_axis))
    x = x_axis.nodes[:, None]
    y = y_axis.nodes[None, :]
    scalar = jnp.sin(2.0 * jnp.pi * x) + 0.5 * jnp.cos(4.0 * jnp.pi * y)
    state = jnp.stack((scalar, 2.0 * scalar), axis=-1)
    expected_scalar = -((2.0 * jnp.pi) ** 2) * jnp.sin(2.0 * jnp.pi * x) - 0.5 * (
        4.0 * jnp.pi
    ) ** 2 * jnp.cos(4.0 * jnp.pi * y)
    expected = jnp.stack((expected_scalar, 2.0 * expected_scalar), axis=-1)

    actual = jax.jit(lambda value: discretization.laplacian(value))(state)

    assert actual.shape == (8, 10, 2)
    assert discretization.flatten(state).shape == (80, 2)
    assert jnp.array_equal(discretization.unflatten(discretization.flatten(state)), state)
    assert jnp.allclose(actual, expected, atol=5e-11)


def test_tensor_eigenpairs_use_exact_separable_modes_and_stable_ordering():
    x_axis = phx.domain.FourierAxisSpec(32).materialize(0.0, 1.0)
    y_axis = phx.domain.CosineAxisSpec(33).materialize(0.0, 1.0)
    discretization = phx.solver.TensorGridDiscretization((x_axis, y_axis))

    eigenvalues, modes = discretization.eigenpairs(rank=8)
    flattened = modes.reshape((discretization.num_points, 8))
    weights = discretization.quadrature_weights.reshape((-1, 1))

    assert modes.shape == (32, 33, 8)
    assert jnp.allclose(
        eigenvalues[:5],
        jnp.asarray([0.0, jnp.pi**2, 4.0 * jnp.pi**2] + [4.0 * jnp.pi**2] * 2),
        atol=1e-12,
    )
    assert jnp.allclose(flattened.T @ (weights * flattened), jnp.eye(8), atol=1e-12)
    assert jnp.allclose(
        discretization.laplacian(modes),
        -modes * eigenvalues,
        atol=2e-9,
    )


def test_tensor_eigenpairs_scale_to_large_product_grids_at_low_rank():
    axes = tuple(phx.domain.FourierAxisSpec(64).materialize(0.0, 1.0) for _ in range(2))
    discretization = phx.solver.TensorGridDiscretization(axes)

    eigenvalues, modes = discretization.eigenpairs(rank=6)

    assert discretization.num_points == 4096
    assert eigenvalues.shape == (6,)
    assert modes.shape == (64, 64, 6)

    long_axis = phx.domain.FourierAxisSpec(10_000).materialize(0.0, 1.0)
    long_grid = phx.solver.TensorGridDiscretization((long_axis,))
    long_eigenvalues, long_modes = long_grid.eigenpairs(rank=4)
    assert long_eigenvalues.shape == (4,)
    assert long_modes.shape == (10_000, 4)


def test_explicit_laplacian_agrees_with_matrix_free_application():
    axis = phx.domain.UniformAxisSpec(
        7,
        endpoint=False,
        periodic=True,
    ).materialize(-1.0, 1.0)
    discretization = phx.solver.TensorGridDiscretization((axis,))
    state = jnp.linspace(-0.4, 0.7, 7)
    matrix = discretization.laplacian_matrix()

    assert matrix.shape == (7, 7)
    assert jnp.allclose(matrix @ state, discretization.laplacian(state))


def test_existing_spectral_plan_is_reused_without_a_second_basis_convention():
    eigenvalues = jnp.asarray([0.0, 1.0, 4.0])
    eigenvectors = jnp.eye(3)
    plan = spectral.SpectralDiscretization.from_eigenpairs(
        eigenvalues,
        eigenvectors,
        jnp.ones((3,)),
        basis_id="unit-plan",
    )
    discretization = phx.solver.SpectralSpatialDiscretization(plan)
    state = jnp.asarray([1.0, 2.0, 3.0])

    assert discretization.plan is plan
    assert discretization.state_shape == (3,)
    assert jnp.allclose(
        discretization.laplacian(state),
        -eigenvalues * state,
    )
    assert jnp.allclose(
        discretization.laplacian_matrix() @ state,
        discretization.laplacian(state),
    )
    retained_values, retained_modes = discretization.eigenpairs(rank=2)
    assert jnp.array_equal(retained_values, eigenvalues[:2])
    assert retained_modes.shape == (3, 2)


def test_spatial_discretization_rejects_unsupported_grids_and_state_shapes():
    nonperiodic_uniform = phx.domain.UniformAxisSpec(
        6,
        endpoint=True,
        periodic=False,
    ).materialize(0.0, 1.0)
    with pytest.raises(ValueError, match="periodic=True"):
        phx.solver.TensorGridDiscretization((nonperiodic_uniform,))

    axis = phx.domain.FourierAxisSpec(6).materialize(0.0, 1.0)
    discretization = phx.solver.TensorGridDiscretization((axis,))
    with pytest.raises(ValueError, match="begin with tensor-grid shape"):
        discretization.laplacian(jnp.ones((5,)))
    with pytest.raises(ValueError, match="rank must lie"):
        discretization.eigenpairs(rank=7)
