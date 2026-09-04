#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import finite_volume as finite_volume_api


def _periodic(count=6):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [2.0 * jnp.pi, 2.0 * jnp.pi]]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    return discretization, operators, momentum


def _taylor_green(discretization):
    x_faces = discretization.face_centers[0]
    y_faces = discretization.face_centers[1]
    return (
        jnp.sin(x_faces[..., 0]) * jnp.cos(x_faces[..., 1]),
        -jnp.cos(y_faces[..., 0]) * jnp.sin(y_faces[..., 1]),
    )


def _block_until_ready(tree):
    return jax.tree.map(
        lambda value: (
            value.block_until_ready() if isinstance(value, jax.Array) else value
        ),
        tree,
    )


def test_mac_variational_viscosity_zero_action_and_frozen_binding():
    discretization, _, momentum = _periodic()
    action = finite_volume_api.PreparedMACVariationalViscosityAction(momentum)
    velocity = _taylor_green(discretization)
    viscosity = jnp.zeros(discretization.cell_shape)
    stage = momentum.boundaries.homogeneous_stage()

    result = action.evaluate(velocity, viscosity, stage)
    frozen = action.freeze(viscosity, stage)
    frozen_result = frozen.evaluate(velocity)

    for block in (
        result.positive_operator_action,
        result.physical_diffusive_rate,
        result.boundary_affine_action,
        frozen_result.physical_diffusive_rate,
    ):
        assert max(float(jnp.max(jnp.abs(value))) for value in block) == 0.0
    assert result.integrated_dissipation == 0.0
    assert result.integrated_work == 0.0
    assert result.operator_row_sum_bound == 0.0
    assert jnp.isinf(result.explicit_step_bound)
    assert result.restriction_supported
    assert result.successful
    assert frozen.prepared_action is action


def test_mac_variational_viscosity_has_positive_work_and_periodic_laplacian_limit():
    discretization, operators, momentum = _periodic(8)
    action = finite_volume_api.PreparedMACVariationalViscosityAction(momentum)
    velocity = _taylor_green(discretization)
    viscosity = jnp.ones(discretization.cell_shape)
    result = action.evaluate(velocity, viscosity, momentum.boundaries.homogeneous_stage())
    laplacian = momentum.homogeneous_laplacian(velocity)

    assert jnp.max(jnp.abs(operators.divergence(velocity))) < 2.0e-12
    for positive, diffusion in zip(
        result.positive_operator_action, laplacian, strict=True
    ):
        np.testing.assert_allclose(positive, -diffusion, rtol=2.0e-12, atol=2.0e-12)
    assert result.positive_work > 0.0
    assert result.integrated_dissipation > 0.0
    np.testing.assert_allclose(
        result.integrated_work,
        -result.integrated_dissipation,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert result.variational_defect < 2.0e-12
    assert result.operator_row_sum_bound > 0.0
    assert result.successful


def test_mac_variational_viscosity_runtime_coefficient_is_jittable_and_has_jvp():
    discretization, operators, momentum = _periodic(5)
    action = finite_volume_api.PreparedMACVariationalViscosityAction(momentum)
    velocity = tuple(
        jnp.sin(points[..., 0] + 2.0 * points[..., 1])
        for points in discretization.face_centers
    )
    viscosity = 1.0 + jnp.arange(25, dtype=jnp.float64).reshape(5, 5) / 25.0
    stage = momentum.boundaries.homogeneous_stage()

    def runtime_rate(coefficient):
        result = action.evaluate(velocity, coefficient, stage)
        return operators.velocity_space.flatten(result.physical_diffusive_rate)

    compiled = jax.jit(runtime_rate)
    eager = runtime_rate(viscosity)
    first = compiled(viscosity)
    second = compiled(2.0 * viscosity)
    _, tangent = jax.jvp(runtime_rate, (viscosity,), (jnp.full_like(viscosity, 0.1),))

    np.testing.assert_allclose(first, eager, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(second, 2.0 * first, rtol=2.0e-12, atol=2.0e-12)
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.linalg.norm(tangent) > 0.0


def test_mac_variational_viscosity_reports_boundary_affine_work():
    count = 6
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    zero_wall = phx.discretization.MACBoundaryProvider(jnp.zeros(2))
    boundaries = phx.discretization.MACBoundaryPlan(
        operators,
        (
            phx.discretization.MACBoundarySide(
                "y", "lower", "no-slip", provider=zero_wall
            ),
            phx.discretization.MACBoundarySide(
                "y",
                "upper",
                "no-slip",
                provider=phx.discretization.MACBoundaryProvider(jnp.asarray([1.0, 0.0])),
            ),
        ),
    ).prepare()
    momentum = phx.discretization.MACMomentumPlan(
        operators, boundaries=boundaries
    ).prepare()
    action = finite_volume_api.PreparedMACVariationalViscosityAction(momentum)
    velocity = tuple(jnp.zeros(layout.shape) for layout in discretization.face_layouts)
    viscosity = jnp.ones(discretization.cell_shape)
    stage = boundaries.evaluate(0.0)

    result = action.evaluate(velocity, viscosity, stage)
    frozen = action.freeze(viscosity, stage)

    assert (
        max(float(jnp.max(jnp.abs(value))) for value in result.positive_operator_action)
        == 0.0
    )
    assert (
        max(float(jnp.max(jnp.abs(value))) for value in result.boundary_affine_action)
        > 0.0
    )
    for direct, bound in zip(
        result.physical_diffusive_rate,
        frozen.physical_diffusive_rate(velocity),
        strict=True,
    ):
        np.testing.assert_allclose(direct, bound, rtol=0.0, atol=0.0)
    for direct, bound in zip(
        result.boundary_affine_action,
        frozen.boundary_affine_action(),
        strict=True,
    ):
        np.testing.assert_allclose(direct, bound, rtol=0.0, atol=0.0)
    assert result.integrated_dissipation > 0.0
    assert result.integrated_work == 0.0
    np.testing.assert_allclose(
        result.boundary_power, result.integrated_dissipation, rtol=0.0, atol=2.0e-12
    )
    assert not result.restriction_supported
    assert jnp.isinf(result.operator_row_sum_bound)
    assert result.successful


@pytest.mark.parametrize("invalid", [-1.0, jnp.inf, jnp.nan])
def test_mac_variational_viscosity_rejects_invalid_runtime_coefficients(invalid):
    discretization, _, momentum = _periodic(4)
    action = finite_volume_api.PreparedMACVariationalViscosityAction(momentum)
    velocity = _taylor_green(discretization)
    viscosity = jnp.full(discretization.cell_shape, invalid)

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="finite and nonnegative"
    ):
        _block_until_ready(action.positive_operator_action(velocity, viscosity))


def test_variable_viscosity_stage_plan_delegates_to_prepared_action():
    discretization, _, momentum = _periodic(4)
    velocity = _taylor_green(discretization)
    density = tuple(jnp.ones_like(value) for value in velocity)
    viscosity = 1.0 + jnp.arange(16, dtype=jnp.float64).reshape(4, 4) / 16.0
    coefficient = jnp.asarray(0.03)

    plan = phx.solver.MACVariableViscosityStagePlan(
        momentum,
        density,
        viscosity,
        coefficient,
        stage_id="variational-viscosity-stage",
    )
    direct = plan.viscosity_action.positive_operator_action(velocity, viscosity)
    applied = plan.momentum_operator.mv(velocity)

    assert isinstance(
        plan.viscosity_action,
        finite_volume_api.PreparedMACVariationalViscosityAction,
    )
    assert isinstance(
        plan.frozen_viscosity_action,
        finite_volume_api.FrozenMACVariationalViscosityAction,
    )
    for result, value, viscous in zip(applied, velocity, direct, strict=True):
        np.testing.assert_allclose(
            result, value + coefficient * viscous, rtol=2.0e-12, atol=2.0e-12
        )
    inverse = plan.inverse(momentum.boundaries.homogeneous_stage())
    assert inverse.momentum_operator is plan.momentum_operator
    assert inverse.stage_id == plan.stage_id

    zero = phx.solver.MACVariableViscosityStagePlan(
        momentum,
        density,
        jnp.zeros(discretization.cell_shape),
        coefficient,
        stage_id="zero-viscosity-stage",
    )
    for result, value in zip(zero.momentum_operator.mv(velocity), velocity, strict=True):
        np.testing.assert_allclose(result, value, rtol=0.0, atol=0.0)

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="finite and nonnegative"
    ):
        invalid = phx.solver.MACVariableViscosityStagePlan(
            momentum,
            density,
            viscosity,
            -0.1,
            stage_id="invalid-viscosity-stage",
        )
        _block_until_ready(invalid.stage_coefficient)
