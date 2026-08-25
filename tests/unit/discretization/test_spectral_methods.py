import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._polynomial._orthogonal import standard_series_value


def _fourier(count=16):
    return phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),),
        axis_names=("x",),
        field_name="u",
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def test_fourier_modal_roundtrip_derivative_parseval_and_gradient():
    space = _fourier(16)
    x = space.axes[0].nodes
    values = 0.3 + jnp.sin(2.0 * jnp.pi * x) - 0.2 * jnp.cos(6.0 * jnp.pi * x)
    coefficients = space.project(values)

    reconstructed = space.reconstruct(coefficients)
    derivative = space.derivative_values(coefficients, axis=0)
    expected = 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * x) + 1.2 * jnp.pi * jnp.sin(
        6.0 * jnp.pi * x
    )
    physical_norm = jnp.sum(space.quadrature_weights * values**2)
    modal_norm = jnp.sum(jnp.abs(coefficients) ** 2)
    gradient = jax.grad(
        lambda data: jnp.sum(space.reconstruct(space.project(data)) ** 2)
    )(values)

    assert jnp.allclose(reconstructed, values, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(derivative, expected, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(physical_norm, modal_norm, rtol=1e-11, atol=1e-11)
    assert jnp.all(jnp.isfinite(gradient))


def test_tensor_spectral_noise_basis_tracks_modal_and_point_value_spaces():
    space = _fourier(8)
    modal = phx.stochastic.SpatialNoiseBasis.from_spectrum(space, 0.1, rank=3)
    refined_modal = phx.stochastic.SpatialNoiseBasis.from_spectrum(
        _fourier(12), 0.1, rank=3
    )
    bounded = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.ConstrainedBasisPlan(
                phx.discretization.LegendreBasisPlan(8),
                phx.discretization.SpectralBoundaryConditionPlan.dirichlet(),
            ),
        ),
        axis_names=("x",),
    ).prepare(jnp.asarray([[-1.0], [1.0]]))
    physical = phx.stochastic.SpatialNoiseBasis.from_kernel_covariance(
        lambda left, right: jnp.exp(-jnp.sum((left - right) ** 2)),
        bounded,
        rank=2,
    )

    assert modal.field_space_id == space.modal_space.field_space_id
    assert modal.mode_ids == refined_modal.mode_ids
    assert modal.mode_storage_dtype == "complex128"
    assert jnp.max(jnp.abs(jnp.imag(modal.modes))) > 0.0
    assert space.imaginary_leakage(modal.modes) < 1e-12
    flattened_modes = modal.modes.reshape((-1, modal.rank))
    assert jnp.allclose(
        flattened_modes.conj().T @ flattened_modes,
        jnp.eye(modal.rank),
        atol=1e-12,
    )
    assert physical.field_space_id == bounded.physical_space.field_space_id
    assert modal.state_shape == space.modal_shape
    assert physical.state_shape == bounded.physical_shape
    assert bounded.modal_shape != bounded.physical_shape


def test_padding_dealiasing_removes_near_cutoff_quadratic_alias():
    space = _fourier(12)
    x = space.axes[0].nodes
    coefficients = space.project(jnp.sin(10.0 * jnp.pi * x))
    dealiasing = phx.discretization.PaddingDealiasingPlan(2).prepare(
        space,
        required_polynomial_degree=2,
    )
    method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2)
    ).prepare(
        space,
        required_polynomial_degree=2,
        nonlinear=True,
    )

    product = dealiasing.project(dealiasing.reconstruct(coefficients) ** 2)
    reconstructed = space.reconstruct(product)
    method_product = method.nonlinear_action(coefficients, lambda value: value**2)

    assert dealiasing.report.evaluation_shape == (18,)
    assert dealiasing.report.exact
    assert jnp.allclose(reconstructed, 0.5, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(method_product, product, rtol=1e-11, atol=1e-11)


def test_chebyshev_and_legendre_derivatives_are_polynomial_exact():
    for basis in (
        phx.discretization.ChebyshevBasisPlan(12),
        phx.discretization.LegendreBasisPlan(12),
    ):
        space = phx.discretization.TensorSpectralPlan(
            (basis,),
            axis_names=("x",),
        ).prepare(jnp.asarray([[-1.0], [1.0]]))
        x = space.axes[0].nodes
        values = x**5 - 2.0 * x**3 + x
        expected = 5.0 * x**4 - 6.0 * x**2 + 1.0

        actual = space.derivative_values(space.project(values), axis=0)

        assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_constrained_legendre_basis_and_boundary_lift_satisfy_endpoint_data():
    boundary = phx.discretization.SpectralBoundaryConditionPlan.dirichlet()
    base_plan = phx.discretization.LegendreBasisPlan(12)
    constrained = phx.discretization.ConstrainedBasisPlan(base_plan, boundary)
    space = phx.discretization.TensorSpectralPlan(
        (constrained,),
        axis_names=("x",),
    ).prepare(jnp.asarray([[-1.0], [1.0]]))
    coefficients = jnp.arange(space.num_modes, dtype=jnp.complex128) / 10.0
    base = base_plan.prepare(-1.0, 1.0, precision=space.plan.precision)
    synthesis = np.asarray(space.axes[0].modal_transform.synthesis)
    base_analysis = np.asarray(base.modal_transform.analysis)
    normalizers = np.sqrt(0.5 * (2.0 * np.arange(base.mode_count) + 1.0))
    base_coefficients = (
        base_analysis @ synthesis @ np.asarray(coefficients)
    ) * normalizers
    lower = standard_series_value("legendre", base_coefficients, jnp.asarray(-1.0))
    upper = standard_series_value("legendre", base_coefficients, jnp.asarray(1.0))
    lift = phx.discretization.BoundaryLiftPlan(
        boundary,
        jnp.asarray((1.0, -2.0)),
    ).prepare(base)
    lift_coefficients = np.asarray(lift.coefficients) * normalizers
    lift_lower = standard_series_value("legendre", lift_coefficients, jnp.asarray(-1.0))
    lift_upper = standard_series_value("legendre", lift_coefficients, jnp.asarray(1.0))

    assert jnp.allclose(lower, 0.0, atol=1e-10)
    assert jnp.allclose(upper, 0.0, atol=1e-10)
    assert jnp.allclose(lift_lower, 1.0, atol=1e-10)
    assert jnp.allclose(lift_upper, -2.0, atol=1e-10)


def test_galerkin_poisson_and_generalized_tau_use_internal_linalg():
    boundary = phx.discretization.SpectralBoundaryConditionPlan.dirichlet()
    space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.ConstrainedBasisPlan(
                phx.discretization.LegendreBasisPlan(12),
                boundary,
            ),
        ),
        axis_names=("x",),
    ).prepare(jnp.asarray([[-1.0], [1.0]]))
    galerkin = phx.discretization.SpectralGalerkinMethodPlan().prepare(space)
    rhs = jnp.ones(space.physical_shape)
    coefficients, linear_result = galerkin.solve_poisson(rhs)

    assert linear_result.successful
    assert (
        jnp.linalg.norm(galerkin.stiffness_action(coefficients) - galerkin.load(rhs))
        < 1e-10
    )

    identity = phx.linalg.DenseLinearOperator(jnp.eye(2))
    tau = phx.discretization.GeneralizedTauPlan(
        identity,
        jnp.asarray([[1.0, 1.0]]),
        jnp.asarray([[1.0], [1.0]]),
    ).prepare()
    result = tau.solve(jnp.zeros((2,)), jnp.asarray([2.0]))

    assert result.linear_result.successful
    assert jnp.allclose(result.field, 1.0, atol=1e-10)
    assert jnp.allclose(result.tau, -1.0, atol=1e-10)
