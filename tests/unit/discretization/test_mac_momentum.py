#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _periodic(count=8):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [2.0 * jnp.pi, 2.0 * jnp.pi]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    return finite_volume, operators, momentum


def _taylor_green(discretization):
    x_faces = discretization.face_centers[0]
    y_faces = discretization.face_centers[1]
    return (
        jnp.sin(x_faces[..., 0]) * jnp.cos(x_faces[..., 1]),
        -jnp.cos(y_faces[..., 0]) * jnp.sin(y_faces[..., 1]),
    )


def test_mac_momentum_certifies_weighted_skew_and_dissipative_diffusion():
    discretization, operators, momentum = _periodic()
    velocity = _taylor_green(discretization)
    convection = momentum.convection(velocity)
    diffusion = momentum.homogeneous_laplacian(velocity)
    space = operators.velocity_space

    assert momentum.report.passed
    assert momentum.report.weighted_skew_residual < 2e-10
    assert momentum.report.diffusion_symmetry_residual < 2e-10
    assert momentum.report.homogeneous_diffusion_rate <= 1e-10
    assert jnp.linalg.norm(operators.divergence(velocity)) < 2e-12
    assert jnp.abs(space.inner(velocity, convection)) < 2e-10
    assert jnp.real(space.inner(velocity, diffusion)) < 0.0


def test_mac_momentum_is_jittable_and_differentiable_in_flat_coordinates():
    discretization, operators, momentum = _periodic(6)
    initial = operators.velocity_space.flatten(_taylor_green(discretization))

    def objective(coordinates):
        velocity = operators.velocity_space.unflatten(coordinates)
        convection = momentum.convection(tuple(velocity))
        rate = operators.velocity_space.flatten(convection)
        return 0.5 * jnp.vdot(rate, rate).real

    eager = objective(initial)
    compiled = jax.jit(objective)(initial)
    gradient = jax.jit(jax.grad(objective))(initial)

    np.testing.assert_allclose(compiled, eager, rtol=1e-12, atol=1e-12)
    assert jnp.all(jnp.isfinite(gradient))


def test_mac_moving_wall_couette_profile_has_zero_momentum_rate():
    count = 8
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    walls = phx.discretization.MACBoundaryPlan(
        operators,
        (
            phx.discretization.MACBoundarySide(
                "y",
                "lower",
                "no-slip",
                provider=phx.discretization.MACBoundaryProvider(jnp.asarray([0.0, 0.0])),
            ),
            phx.discretization.MACBoundarySide(
                "y",
                "upper",
                "no-slip",
                provider=phx.discretization.MACBoundaryProvider(jnp.asarray([1.0, 0.0])),
            ),
        ),
    )
    momentum = phx.discretization.MACMomentumPlan(operators, boundaries=walls).prepare()
    y = finite_volume.face_centers[0][..., 1]
    velocity = (y, jnp.zeros(finite_volume.face_layouts[1].shape))

    convection = momentum.convection(velocity)
    diffusion = momentum.laplacian(velocity)
    diagnostics = momentum.diagnostics(velocity)

    assert diagnostics.boundary_defect < 1e-13
    assert max(float(jnp.max(jnp.abs(value))) for value in convection) < 2e-12
    assert max(float(jnp.max(jnp.abs(value))) for value in diffusion) < 2e-11


def test_mac_wall_boundaries_report_normal_flux_incompatibility():
    bounded_grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4),
            phx.discretization.UniformCellAxisSpec(4),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    bounded = phx.discretization.MACOperatorPlan(
        phx.discretization.FiniteVolumePlan(bounded_grid).prepare()
    ).prepare()
    zero = phx.discretization.MACBoundaryProvider(jnp.zeros(2))
    boundaries = phx.discretization.MACBoundaryPlan(
        bounded,
        (
            phx.discretization.MACBoundarySide(
                "x",
                "lower",
                "normal-flux-inflow",
                provider=phx.discretization.MACBoundaryProvider(0.1),
            ),
            phx.discretization.MACBoundarySide("x", "upper", "no-slip", provider=zero),
            phx.discretization.MACBoundarySide("y", "lower", "no-slip", provider=zero),
            phx.discretization.MACBoundarySide("y", "upper", "no-slip", provider=zero),
        ),
    ).prepare()

    stage = boundaries.evaluate(0.0)

    assert not stage.successful
    assert jnp.abs(stage.compatibility_defect) > 0.0


def test_three_dimensional_mac_constant_velocity_has_zero_rate():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(4, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    velocity = tuple(
        jnp.full(layout.shape, 0.2 * (axis + 1))
        for axis, layout in enumerate(finite_volume.face_layouts)
    )

    convection = momentum.convection(velocity)
    diffusion = momentum.laplacian(velocity)

    assert operators.velocity_space.size == sum(value.size for value in velocity)
    assert max(float(jnp.max(jnp.abs(value))) for value in convection) < 2e-12
    assert max(float(jnp.max(jnp.abs(value))) for value in diffusion) < 2e-12
