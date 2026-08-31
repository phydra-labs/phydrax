import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _fourier_space(count=12):
    return phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),),
        axis_names=("x",),
        field_name="u",
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))


def _quadratic_problem(*, condition=False):
    x = phx.equations.PDECoordinate(
        "x",
        "space",
        bounds=(0.0, 1.0),
        periodic=True,
    )
    field = phx.equations.PDEField("u", coordinates=("x",))
    u = phx.equations.PDEExpression.field("u")
    regions = (
        (phx.equations.PDERegion("boundary", "boundary", ("x",)),) if condition else ()
    )
    conditions = (
        (
            phx.equations.PDECondition(
                "boundary",
                "boundary",
                u,
                region="boundary",
                coordinate="x",
            ),
        )
        if condition
        else ()
    )
    return phx.equations.PDEProblemIR(
        (x,),
        (field,),
        equations=(
            phx.equations.PDEEquation(
                "quadratic",
                u * u,
                phx.equations.PDEExpression.constant(0.5),
            ),
        ),
        conditions=conditions,
        regions=regions,
    )


def test_full_closure_detects_residual_outside_retained_projection():
    space = _fourier_space()
    x = space.axes[0].nodes
    wave_number = space.modal_shape[0] // 2 - 1
    state = space.project(jnp.sin(2.0 * jnp.pi * wave_number * x))
    retained = phx.equations.compile_spectral_residual(
        _quadratic_problem(),
        space,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PaddingDealiasingPlan(2),
        ),
        scope="retained",
    )
    full = phx.equations.compile_spectral_residual(
        _quadratic_problem(),
        space,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PolynomialClosureDealiasingPlan(2),
        ),
    )

    assert retained.residual_energy(state) < 1e-20
    assert jnp.allclose(full.residual_energy(state), 0.125, rtol=1e-11, atol=1e-11)
    assert full.report.exact
    assert full.report.evaluation_shape == (23,)
    assert full.residual_coefficients(state)[0].shape == (23,)


def test_all_coordinate_chebyshev_time_derivative_is_exact_and_differentiable():
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("t",))
    u = phx.equations.PDEExpression.field("u")
    problem = phx.equations.PDEProblemIR(
        (t,),
        (field,),
        equations=(
            phx.equations.PDEEquation(
                "unit-rate",
                u.derivative("t"),
                phx.equations.PDEExpression.constant(1.0),
            ),
        ),
    )
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.ChebyshevBasisPlan(8),),
        axis_names=("t",),
        field_name="u",
    ).prepare((phx.discretization.AxisDomain.interval(0.0, 1.0),))
    compiled = phx.equations.compile_spectral_residual(
        problem,
        space,
        phx.discretization.PseudospectralMethodPlan(),
    )
    state = compiled.project_state(space.axes[0].nodes)
    energy = jax.jit(lambda value: compiled.residual_energy(value))(state)
    gradient = jax.grad(lambda value: compiled.residual_energy(value))(state)

    assert energy < 1e-20
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.sqrt(jnp.real(jnp.vdot(gradient, gradient))) < 1e-8


def test_nonpolynomial_full_residual_requires_explicit_approximation():
    space = _fourier_space(8)
    problem = _quadratic_problem()
    u = phx.equations.PDEExpression.field("u")
    problem = phx.equations.PDEProblemIR(
        problem.coordinates,
        problem.fields,
        equations=(phx.equations.PDEEquation("exponential", u.exp()),),
    )
    method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.ModalFilterPlan(),
    )

    with pytest.raises(ValueError, match="cannot certify"):
        phx.equations.compile_spectral_residual(problem, space, method)

    compiled = phx.equations.compile_spectral_residual(
        problem,
        space,
        method,
        require_exact=False,
    )
    assert not compiled.report.exact


def test_conditions_are_rejected_unless_external_handling_is_explicit():
    space = _fourier_space(8)
    problem = _quadratic_problem(condition=True)
    method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PolynomialClosureDealiasingPlan(2),
    )

    with pytest.raises(ValueError, match="external hard-condition"):
        phx.equations.compile_spectral_residual(problem, space, method)

    compiled = phx.equations.compile_spectral_residual(
        problem,
        space,
        method,
        condition_handling="external",
    )
    assert compiled.report.condition_handling == "external"


def test_closure_plan_rejects_unsupported_capacity_and_basis():
    space = _fourier_space(12)
    with pytest.raises(ValueError, match="maximum_evaluation_modes"):
        phx.discretization.PolynomialClosureDealiasingPlan(
            2,
            maximum_evaluation_modes=16,
        ).prepare(space, required_polynomial_degree=2)

    sine = phx.discretization.TensorSpectralPlan(
        (phx.discretization.SineBasisPlan(8),),
        axis_names=("x",),
    ).prepare((phx.discretization.AxisDomain.interval(0.0, 1.0),))
    with pytest.raises(ValueError, match="sine basis"):
        phx.discretization.PolynomialClosureDealiasingPlan(2).prepare(
            sine,
            required_polynomial_degree=2,
        )


def test_periodic_coordinate_values_do_not_claim_finite_exactness():
    space = _fourier_space(8)
    x = phx.equations.PDECoordinate(
        "x",
        "space",
        bounds=(0.0, 1.0),
        periodic=True,
    )
    field = phx.equations.PDEField("u", coordinates=("x",))
    u = phx.equations.PDEExpression.field("u")
    coordinate = phx.equations.PDEExpression.coordinate_value("x")
    problem = phx.equations.PDEProblemIR(
        (x,),
        (field,),
        equations=(phx.equations.PDEEquation("coordinate", u, coordinate),),
    )

    with pytest.raises(ValueError, match="cannot certify"):
        phx.equations.compile_spectral_residual(
            problem,
            space,
            phx.discretization.PseudospectralMethodPlan(),
        )

    compiled = phx.equations.compile_spectral_residual(
        problem,
        space,
        phx.discretization.PseudospectralMethodPlan(),
        require_exact=False,
    )
    assert not compiled.report.exact
