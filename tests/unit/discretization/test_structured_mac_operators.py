import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _prepared(count=24, *, periodic=True):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count, periodic=periodic),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    return grid, phx.discretization.MACOperatorPlan(finite_volume).prepare()


def test_mac_operator_bundle_certifies_adjoint_nullspace_and_laplacian():
    grid, operators = _prepared(32)
    pressure = jnp.sin(2.0 * jnp.pi * grid.structured_axes[0].interval_centers)
    gradient = operators.gradient(pressure)
    laplacian = operators.laplacian(pressure)

    assert operators.report.passed
    assert operators.report.transform_eligible
    assert operators.report.weighted_adjoint_residual < 1e-12
    assert operators.report.constant_laplacian_residual < 1e-12
    np.testing.assert_allclose(operators.divergence(gradient), laplacian, atol=1e-13)
    np.testing.assert_allclose(
        jnp.sum(
            operators.discretization.cell_volumes * operators.gauge_project(pressure)
        ),
        0.0,
        atol=2e-12,
    )


def test_mac_velocity_block_space_preserves_face_pairing_and_coordinates():
    _, operators = _prepared(16)
    velocity = (jnp.sin(2.0 * jnp.pi * jnp.arange(16) / 16.0),)

    coordinates = operators.velocity_space.flatten(velocity)
    restored = operators.velocity_space.unflatten(coordinates)
    energy = operators.velocity_space.inner(velocity, velocity)
    expected = jnp.sum(operators.face_dual_measures[0] * velocity[0] ** 2)

    assert coordinates.shape == (16,)
    np.testing.assert_allclose(restored[0], velocity[0], atol=0.0)
    np.testing.assert_allclose(energy, expected, atol=1e-14)


def test_mac_rate_projection_uses_the_same_transform_and_iterative_actions():
    _, operators = _prepared(32)
    rate = (jnp.sin(2.0 * jnp.pi * jnp.arange(32) / 32.0),)

    transform = phx.solver.MACPressureProjectionPlan(
        operators, solve_method="transform", tolerance=1e-10
    ).project_rate(rate)
    iterative = phx.solver.MACPressureProjectionPlan(
        operators, solve_method="iterative", tolerance=1e-10
    ).project_rate(rate)

    assert transform.converged
    assert iterative.converged
    np.testing.assert_allclose(transform.rate[0], iterative.rate[0], rtol=0.0, atol=2e-8)
    assert jnp.linalg.norm(transform.divergence_after) < 1e-9
    assert jnp.linalg.norm(iterative.divergence_after) < 1e-8


def test_mac_projection_transform_and_iterative_routes_agree():
    _, operators = _prepared(32)
    velocity = (jnp.sin(2.0 * jnp.pi * jnp.arange(32) / 32.0),)
    transform = phx.solver.MACPressureProjectionPlan(
        operators, solve_method="transform", tolerance=1e-10
    ).project(velocity, 0.1)
    iterative = phx.solver.MACPressureProjectionPlan(
        operators, solve_method="iterative", tolerance=1e-10
    ).project(velocity, 0.1)

    assert transform.converged
    assert iterative.converged
    assert transform.solve_method == "transform"
    assert iterative.solve_method == "iterative"
    np.testing.assert_allclose(
        transform.velocity[0], iterative.velocity[0], rtol=0.0, atol=2e-8
    )
    assert jnp.linalg.norm(transform.divergence_after) < 1e-9
    assert jnp.linalg.norm(iterative.divergence_after) < 1e-8


def test_mac_variable_coefficient_projection_is_jittable_and_idempotent():
    grid, operators = _prepared(24)
    projection = phx.solver.MACPressureProjectionPlan(
        operators, solve_method="iterative", tolerance=1e-9
    )
    velocity = (jnp.cos(2.0 * jnp.pi * jnp.arange(24) / 24.0),)
    centers = grid.structured_axes[0].interval_centers
    inverse = 0.1 * (1.0 + 0.2 * jnp.cos(2.0 * jnp.pi * centers))
    first = jax.jit(
        lambda current, coefficient: projection.project(
            current,
            0.1,
            inverse_momentum_diagonal=coefficient,
        )
    )(velocity, inverse)
    second = projection.project(
        first.velocity,
        0.1,
        pressure=first.pressure,
        inverse_momentum_diagonal=inverse,
    )

    assert first.converged
    assert second.converged
    assert jnp.linalg.norm(first.divergence_after) < 1e-7
    np.testing.assert_allclose(second.velocity[0], first.velocity[0], atol=2e-7)


def test_mac_projection_validates_coefficients_and_transform_eligibility():
    _, operators = _prepared(12)
    transform = phx.solver.MACPressureProjectionPlan(operators, solve_method="transform")
    velocity = (jnp.ones((12,)),)
    with pytest.raises(ValueError, match="variable"):
        transform.project(
            velocity,
            0.1,
            inverse_momentum_diagonal=jnp.ones((12,)),
        )
    with pytest.raises(Exception, match="positive"):
        phx.solver.MACPressureProjectionPlan(operators, solve_method="iterative").project(
            velocity,
            0.1,
            inverse_momentum_diagonal=-jnp.ones((12,)),
        )
    with pytest.raises(Exception, match="step_size"):
        transform.project(velocity, 0.0)
