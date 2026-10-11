import itertools
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._polynomial._orthogonal import standard_series_value


def _fourier(count: Any = 16) -> Any:
    return phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),),
        axis_names=("x",),
        field_name="u",
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))


def test_spectral_methods_scenario_1() -> None:
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
    ).prepare((phx.discretization.AxisDomain.interval(-1.0, 1.0),))
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


def test_spectral_methods_scenario_2() -> None:
    for basis in (
        phx.discretization.ChebyshevBasisPlan(12),
        phx.discretization.LegendreBasisPlan(12),
    ):
        space = phx.discretization.TensorSpectralPlan(
            (basis,),
            axis_names=("x",),
        ).prepare((phx.discretization.AxisDomain.interval(-1.0, 1.0),))
        x = space.axes[0].nodes
        values = x**5 - 2.0 * x**3 + x
        expected = 5.0 * x**4 - 6.0 * x**2 + 1.0

        actual = space.derivative_values(space.project(values), axis=0)

        assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-10)
    boundary = phx.discretization.SpectralBoundaryConditionPlan.dirichlet()
    base_plan = phx.discretization.LegendreBasisPlan(12)
    constrained = phx.discretization.ConstrainedBasisPlan(base_plan, boundary)
    space = phx.discretization.TensorSpectralPlan(
        (constrained,),
        axis_names=("x",),
    ).prepare((phx.discretization.AxisDomain.interval(-1.0, 1.0),))
    coefficients = jnp.arange(space.num_modes, dtype=jnp.complex128) / 10.0
    base = base_plan.prepare(
        phx.discretization.AxisDomain.interval(-1.0, 1.0),
        precision=space.plan.precision,
    )
    # ty: ignore[unresolved-attribute]
    synthesis = np.asarray(space.axes[0].modal_transform.synthesis)
    # ty: ignore[unresolved-attribute]
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
    boundary = phx.discretization.SpectralBoundaryConditionPlan.dirichlet()
    space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.ConstrainedBasisPlan(
                phx.discretization.LegendreBasisPlan(12),
                boundary,
            ),
        ),
        axis_names=("x",),
    ).prepare((phx.discretization.AxisDomain.interval(-1.0, 1.0),))
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


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("source_even", [False, True])
@pytest.mark.parametrize("target_even", [False, True])
def test_retained_band_restriction_signed_mode_oracle(
    rank: int, source_even: bool, target_even: bool
) -> None:
    # Every axis exercises both odd and even counts across parameterizations.
    source_shape = tuple(7 + ((axis + source_even) % 2) for axis in range(rank))
    target_shape = tuple(4 + ((axis + target_even) % 2) for axis in range(rank))

    def space(shape: tuple[int, ...]) -> phx.discretization.TensorSpectralDiscretization:
        return phx.discretization.TensorSpectralPlan(
            tuple(phx.discretization.FourierBasisPlan(n) for n in shape)
        ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),) * rank)

    prepared = phx.discretization.RetainedBandRestrictionPlan(
        space(source_shape), space(target_shape)
    ).prepare()
    coefficients = (
        np.arange(np.prod(source_shape) * 2, dtype=np.float64).reshape(
            source_shape + (2,)
        )
        + 1j
    )
    expected = np.zeros(target_shape + (2,), dtype=np.complex128)
    retained_source = np.zeros(source_shape, dtype=np.bool_)
    retained_count = 0
    for index in itertools.product(*(range(n) for n in target_shape)):
        if any(n % 2 == 0 and i == n // 2 for i, n in zip(index, target_shape)):
            continue
        signed = tuple(
            i if i <= (n - 1) // 2 else i - n for i, n in zip(index, target_shape)
        )
        source_index = tuple(k % n for k, n in zip(signed, source_shape))
        expected[index] = coefficients[source_index]
        retained_source[source_index] = True
        retained_count += 1
    actual = np.asarray(prepared(coefficients))
    np.testing.assert_array_equal(actual, expected)
    discarded = np.sum(np.abs(coefficients[~retained_source]) ** 2)
    np.testing.assert_allclose(
        np.sum(np.abs(coefficients) ** 2) - np.sum(np.abs(actual) ** 2),
        discarded,
        rtol=2e-15,
    )
    assert prepared.retained_mode_count == retained_count
    assert (
        prepared.discarded_forbidden_mode_count == np.prod(source_shape) - retained_count
    )
    for axis, count in enumerate(target_shape):
        if count % 2 == 0:
            np.testing.assert_array_equal(np.take(actual, count // 2, axis=axis), 0)


def test_prepared_retained_band_restriction_is_dynamic_jit_argument() -> None:
    prepared = phx.discretization.RetainedBandRestrictionPlan(
        _fourier(12), _fourier(6)
    ).prepare()

    @eqx.filter_jit
    def apply(
        restriction: phx.discretization.PreparedRetainedBandRestriction,
        values: jax.Array,
    ) -> jax.Array:
        return restriction(values)

    coefficients = jnp.arange(24, dtype=jnp.float64).reshape(12, 2)
    np.testing.assert_array_equal(apply(prepared, coefficients), prepared(coefficients))
    np.testing.assert_array_equal(
        jax.vmap(lambda values: apply(prepared, values))(
            jnp.stack((coefficients, coefficients + 1))
        ),
        np.stack((prepared(coefficients), prepared(coefficients + 1))),
    )


def test_retained_band_restriction_is_transitive() -> None:
    fine, middle, coarse = _fourier(12), _fourier(9), _fourier(6)
    coefficients = jnp.asarray(np.arange(24).reshape(12, 2), dtype=jnp.complex128)
    first = phx.discretization.RetainedBandRestrictionPlan(fine, middle).prepare()
    second = phx.discretization.RetainedBandRestrictionPlan(middle, coarse).prepare()
    direct = phx.discretization.RetainedBandRestrictionPlan(fine, coarse).prepare()
    np.testing.assert_array_equal(second(first(coefficients)), direct(coefficients))
