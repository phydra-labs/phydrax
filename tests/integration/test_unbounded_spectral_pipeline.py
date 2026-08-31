import jax
import jax.numpy as jnp

import phydrax as phx


def test_rational_line_compiles_matrix_free_heat_dynamics_with_gradients():
    x = phx.equations.PDECoordinate("x", "space")
    time = phx.equations.PDECoordinate("t", "time")
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    problem = phx.equations.PDEProblemIR(
        (x, time),
        (field,),
        equations=(
            phx.equations.PDEEquation(
                "heat",
                u.derivative("t"),
                u.laplacian("x"),
            ),
        ),
    )
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.RationalChebyshevLineBasisPlan(12, 2.0),),
        axis_names=("x",),
        field_name="u",
    ).prepare((phx.discretization.AxisDomain.real_line(),))
    compiled = phx.equations.compile_semidiscrete_pde(
        problem,
        space,
        phx.discretization.PseudospectralMethodPlan(),
    )
    physical = jnp.exp(-(space.axes[0].nodes ** 2))
    state = compiled.project_state(physical)
    rate = jax.jit(lambda value: compiled(0.0, value, None))(state)
    expected = space.modal_laplacian(state)
    gradient = jax.grad(
        lambda value: jnp.real(jnp.vdot(compiled(0.0, value, None), value))
    )(state)

    assert compiled.resolved_method == "spectral-semilinear-matrix-free"
    assert jnp.allclose(rate, expected, atol=1e-10)
    assert jnp.all(jnp.isfinite(gradient))
