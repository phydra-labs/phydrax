#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _immersed_system(count=8):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(operators).prepare()
    marker_position = jnp.asarray(
        [[0.35, 0.35], [0.65, 0.35], [0.65, 0.65], [0.35, 0.65]]
    )
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(4), marker_position, jnp.full((4,), 0.25)
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(
        operators, markers
    ).prepare()
    projection = phx.solver.MACImmersedBoundaryProjectionPlan(
        operators,
        transfer,
        boundaries=boundaries,
        tolerance=1.0e-8,
        maximum_iterations=200,
    )
    return finite_volume, operators, boundaries, markers, transfer, projection


def test_exact_immersed_projection_preserves_zero_state():
    finite_volume, operators, _, markers, _, projection = _immersed_system()
    zero_velocity = tuple(
        jnp.zeros(layout.shape) for layout in finite_volume.face_layouts
    )
    kinematics = markers.kinematics(
        markers.reference_position, jnp.zeros_like(markers.reference_position)
    )
    result = projection.project(zero_velocity, 0.01, kinematics)

    assert result.successful
    assert jnp.linalg.norm(result.divergence_after) < 1.0e-8
    assert jnp.linalg.norm(result.marker_slip) < 1.0e-8
    assert result.gauge_defect < 1.0e-8
    assert result.kkt_residual_norm < 1.0e-8


def test_marker_interpolation_adjoint_matches_spread_and_jit():
    finite_volume, operators, _, markers, transfer, _ = _immersed_system()
    relation = transfer.relation(markers.reference_position)
    operator = transfer.interpolation_operator(relation)
    velocity = tuple(
        jnp.sin(jnp.arange(int(jnp.prod(jnp.asarray(layout.shape)))).reshape(layout.shape))
        for layout in finite_volume.face_layouts
    )
    multiplier = jnp.asarray(
        [[0.2, -0.1], [0.3, 0.4], [-0.2, 0.1], [0.1, -0.3]]
    )
    spread = transfer.spread(relation, multiplier)
    adjoint = operator.adjoint_mv(multiplier)
    jitted = jax.jit(lambda value: transfer.gather(relation, value))(velocity)

    assert all(jnp.allclose(left, right) for left, right in zip(spread, adjoint, strict=True))
    assert jnp.allclose(jitted, transfer.gather(relation, velocity))
    assert jnp.isclose(
        operators.velocity_space.inner(velocity, spread),
        markers.active_velocity_space.inner(operator.mv(velocity), multiplier),
        atol=1.0e-10,
    )


def test_marker_geometry_jvp_is_finite_inside_fixed_routes():
    finite_volume, _, _, markers, transfer, _ = _immersed_system()
    velocity = tuple(
        jnp.sin(jnp.arange(int(jnp.prod(jnp.asarray(layout.shape)))).reshape(layout.shape))
        for layout in finite_volume.face_layouts
    )
    position = markers.reference_position + jnp.asarray([0.013, -0.017])
    tangent = jnp.full_like(position, 0.01)

    def observable(value):
        relation = transfer.relation(value)
        return jnp.sum(transfer.gather(relation, velocity) ** 2)

    value, derivative = jax.jvp(observable, (position,), (tangent,))
    epsilon = 1.0e-5
    finite_difference = (
        observable(position + epsilon * tangent)
        - observable(position - epsilon * tangent)
    ) / (2.0 * epsilon)

    assert jnp.isfinite(value)
    assert jnp.isfinite(derivative)
    assert jnp.allclose(derivative, finite_difference, rtol=2.0e-5, atol=2.0e-6)
