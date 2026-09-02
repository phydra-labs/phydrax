import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _reaction_diffusion_problem():
    x = phx.equations.PDECoordinate(
        "x",
        "space",
        bounds=(0.0, 1.0),
        periodic=True,
    )
    t = phx.equations.PDECoordinate("t", "time")
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    return phx.equations.PDEProblemIR(
        (x, t),
        (field,),
        equations=(
            phx.equations.PDEEquation(
                "reaction-diffusion",
                u.derivative("t"),
                0.1 * u.laplacian("x") + u * (1.0 - u),
            ),
        ),
    )


def _space(count=24):
    return phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),),
        axis_names=("x",),
        field_name="u",
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))


def test_spectral_compiler_requires_dealiasing_for_nonlinearity():
    with pytest.raises(ValueError, match="requires an explicit dealiasing"):
        phx.equations.compile_semidiscrete_pde(
            _reaction_diffusion_problem(),
            _space(),
            phx.discretization.PseudospectralMethodPlan(),
        )


def test_spectral_compiler_matches_physical_reference_jit_and_gradient():
    space = _space()
    method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2),
    )
    compiled = phx.equations.compile_semidiscrete_pde(
        _reaction_diffusion_problem(),
        space,
        method,
    )
    x = space.axes[0].nodes
    physical = 0.2 + 0.1 * jnp.sin(2.0 * jnp.pi * x)
    state = compiled.project_state(physical)
    expected = 0.1 * space.laplacian(physical) + physical * (1.0 - physical)

    modal_rate = compiled(0.0, state, None)
    physical_rate = compiled.reconstruct_state(modal_rate)
    jitted = jax.jit(lambda value: compiled(0.0, value, None))(state)
    gradient = jax.grad(
        lambda value: jnp.real(jnp.vdot(compiled(0.0, value, None), value))
    )(state)

    assert compiled.resolved_method == "spectral-semilinear-diagonal"
    assert isinstance(
        compiled.semilinear_drift.linear_operator,
        phx.linalg.DiagonalLinearOperator,
    )
    assert jnp.allclose(physical_rate, expected, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(jitted, modal_rate, rtol=1e-11, atol=1e-11)
    assert jnp.all(jnp.isfinite(gradient))


def test_spherical_heat_compiles_to_coefficient_resident_diagonal_dynamics():
    sphere_coordinate = phx.equations.PDECoordinate("sphere", "space", size=2)
    time = phx.equations.PDECoordinate("t", "time")
    field = phx.equations.PDEField("u", coordinates=("sphere", "t"))
    u = phx.equations.PDEExpression.field("u")
    problem = phx.equations.PDEProblemIR(
        (sphere_coordinate, time),
        (field,),
        equations=(
            phx.equations.PDEEquation(
                "spherical-heat",
                u.derivative("t"),
                0.2 * u.laplacian("sphere"),
            ),
        ),
    )
    space = phx.discretization.SphericalSpectralPlan(5).prepare(radius=1.7)
    compiled = phx.equations.compile_semidiscrete_pde(
        problem,
        space,
        phx.discretization.PseudospectralMethodPlan(),
    )
    coefficients = jnp.zeros(space.coefficient_shape, dtype=jnp.complex128)
    coefficients = coefficients.at[3, 6].set(0.4 - 0.2j)
    coefficients = space.layout.canonicalize_reality(coefficients)
    rate = compiled(0.0, coefficients, None)
    expected = 0.2 * space.modal_laplacian(coefficients)

    assert compiled.resolved_method == "spectral-semilinear-diagonal"
    assert jnp.allclose(rate, expected, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(space.modal_integral(rate), 0.0, atol=1e-12)
