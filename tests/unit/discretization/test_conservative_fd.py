#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _cell_grid(points=64, *, periodic=False):
    return phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(
                points,
                periodic=periodic,
            ),
        ),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def _dirichlet_boundaries():
    return {"x": ("dirichlet", "dirichlet")}


def test_harmonic_face_interpolation_preserves_discontinuous_material_flux():
    grid = _cell_grid(64)
    x = grid.axes[0].nodes
    coefficient = jnp.where(x < 0.5, 1.0, 10.0)
    operator = phx.discretization.ConservativeDiffusionPlan(
        grid,
        boundaries=_dirichlet_boundaries(),
        interpolation="harmonic",
    ).prepare(coefficient)
    flux = 1.0 / (0.5 / 1.0 + 0.5 / 10.0)
    exact = jnp.where(
        x < 0.5,
        flux * x,
        flux * 0.5 + flux * (x - 0.5) / 10.0,
    )

    action = operator.apply(exact, boundary_values={"x": (0.0, 1.0)})
    face_flux = operator.fluxes(
        exact,
        coefficient,
        {"x": (0.0, 1.0)},
    )[0]

    np.testing.assert_allclose(action, 0.0, rtol=0.0, atol=3e-11)
    np.testing.assert_allclose(face_flux, flux, rtol=2e-12, atol=2e-12)


def test_neumann_diffusion_is_globally_conservative_and_has_weighted_adjoint():
    grid = _cell_grid(47)
    x = grid.axes[0].nodes
    coefficient = 1.0 + x**2
    operator = phx.discretization.ConservativeDiffusionPlan(grid).prepare(coefficient)
    state = jnp.sin(3.0 * jnp.pi * x) + 0.2 * jnp.cos(5.0 * jnp.pi * x)
    probe = jnp.cos(2.0 * jnp.pi * x)

    action = operator.mv(state)
    left = jnp.sum(grid.quadrature_weights * probe * action)
    right = jnp.sum(grid.quadrature_weights * operator.adjoint_mv(probe) * state)

    assert operator.conservation_report.conservative
    assert operator.stability_report.passed
    np.testing.assert_allclose(
        jnp.sum(grid.quadrature_weights * action),
        0.0,
        rtol=0.0,
        atol=2e-11,
    )
    np.testing.assert_allclose(left, right, rtol=2e-10, atol=2e-10)


def test_full_anisotropic_tensor_diffusion_preserves_conservative_flux_balance():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(24),
            phx.discretization.UniformCellAxisSpec(20),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    tensor = jnp.broadcast_to(
        jnp.asarray([[2.0, 0.5], [0.5, 3.0]]),
        grid.shape + (2, 2),
    )
    operator = phx.discretization.ConservativeDiffusionPlan(grid).prepare(tensor)
    x = grid.axes[0].nodes[:, None]
    y = grid.axes[1].nodes[None, :]
    state = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

    action = operator.mv(state)

    np.testing.assert_allclose(
        jnp.sum(grid.quadrature_weights * action),
        0.0,
        rtol=0.0,
        atol=2e-10,
    )
    assert operator.stability_report.passed is None


def test_periodic_conservative_and_skew_advection_preserve_mass_and_energy():
    grid = _cell_grid(96, periodic=True)
    velocity = (jnp.ones(grid.faces("x").shape),)
    x = grid.axes[0].nodes
    state = 0.4 + jnp.sin(2.0 * jnp.pi * x) + 0.2 * jnp.cos(6.0 * jnp.pi * x)
    conservative = phx.discretization.ConservativeAdvectionPlan(
        grid,
        form="conservative",
        reconstruction="upwind",
    ).prepare(velocity)
    skew = phx.discretization.ConservativeAdvectionPlan(
        grid,
        form="skew",
        reconstruction="arithmetic",
    ).prepare(velocity)

    conservative_action = conservative.apply(state)
    skew_action = skew.apply(state)

    np.testing.assert_allclose(
        jnp.sum(grid.quadrature_weights * conservative_action),
        0.0,
        rtol=0.0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        jnp.sum(grid.quadrature_weights * state * skew_action),
        0.0,
        rtol=0.0,
        atol=2e-12,
    )
    assert conservative.plan.plan_id != skew.plan.plan_id


def _conservative_problem(rhs, *, parameters):
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0))
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    u_field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    return phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(u_field,),
        parameters=parameters,
        equations=(phx.equations.PDEEquation("evolution", u.derivative("t"), rhs),),
        conditions=(
            phx.equations.PDECondition(
                "lower",
                "boundary",
                u,
                target=phx.equations.PDEExpression.constant(0.0),
                region="lower",
                coordinate="x",
            ),
            phx.equations.PDECondition(
                "upper",
                "boundary",
                u,
                target=phx.equations.PDEExpression.constant(1.0),
                region="upper",
                coordinate="x",
            ),
        ),
        regions=(
            phx.equations.PDERegion(
                "lower",
                "boundary",
                ("x",),
                component="lower",
            ),
            phx.equations.PDERegion(
                "upper",
                "boundary",
                ("x",),
                component="upper",
            ),
        ),
    )


def test_native_compiler_preserves_conservative_diffusion_expression_form():
    grid = _cell_grid(32)
    u = phx.equations.PDEExpression.field("u")
    coefficient_expression = phx.equations.PDEExpression.parameter("a")
    rhs = (coefficient_expression * u.gradient("x")).divergence("x")
    problem = _conservative_problem(
        rhs,
        parameters=(phx.equations.PDEParameter("a", functional=True),),
    )
    compiled = phx.equations.compile_finite_difference_pde(problem, grid)
    x = grid.axes[0].nodes
    coefficient = 1.0 + x
    state = x**2
    explicit = phx.discretization.ConservativeDiffusionPlan(
        grid,
        boundaries=_dirichlet_boundaries(),
    ).prepare(coefficient)

    compiled_action = compiled(0.0, state, {"a": coefficient})
    explicit_action = explicit.apply(
        state,
        boundary_values={"x": (0.0, 1.0)},
    )

    np.testing.assert_allclose(
        compiled_action,
        explicit_action,
        rtol=2e-12,
        atol=2e-12,
    )


def test_native_compiler_preserves_conservative_advection_expression_form():
    grid = _cell_grid(48)
    u = phx.equations.PDEExpression.field("u")
    velocity_expression = phx.equations.PDEExpression.parameter("velocity")
    rhs = (velocity_expression * u).divergence("x")
    problem = _conservative_problem(
        rhs,
        parameters=(phx.equations.PDEParameter("velocity", functional=True),),
    )
    compiled = phx.equations.compile_finite_difference_pde(problem, grid)
    x = grid.axes[0].nodes
    state = x * (1.0 - x)
    velocity = jnp.ones(grid.shape + (1,))
    explicit = phx.discretization.ConservativeAdvectionPlan(
        grid,
        form="conservative",
        boundaries=_dirichlet_boundaries(),
    ).prepare(velocity)

    compiled_action = compiled(0.0, state, {"velocity": velocity})
    explicit_action = explicit.apply(
        state,
        boundary_values={"x": (0.0, 1.0)},
    )

    np.testing.assert_allclose(
        compiled_action,
        explicit_action,
        rtol=2e-12,
        atol=2e-12,
    )
