#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _wing(panel_count=4):
    span = jnp.linspace(-1.5, 1.5, panel_count + 1)
    leading = jnp.stack((jnp.zeros_like(span), span, jnp.zeros_like(span)), axis=-1)
    trailing = leading + jnp.asarray((1.0, 0.0, 0.0))
    return phx.discretization.LiftingSurfacePlan(leading, trailing).prepare()


def test_regularized_filament_orientation_reverses_velocity():
    targets = jnp.asarray(((0.5, 1.0, 0.0),))
    start = jnp.asarray(((0.0, 0.0, 0.0),))
    end = jnp.asarray(((1.0, 0.0, 0.0),))
    forward = phx.operators.regularized_filament_velocity_3d(
        targets, start, end, jnp.ones((1,)), jnp.full((1,), 0.01)
    )
    reverse = phx.operators.regularized_filament_velocity_3d(
        targets, end, start, jnp.ones((1,)), jnp.full((1,), 0.01)
    )

    np.testing.assert_allclose(reverse, -forward, rtol=1e-12, atol=1e-12)
    assert jnp.linalg.norm(forward) > 0.0


def test_steady_vortex_lattice_solves_impermeability_and_returns_lift():
    surface = _wing(6)
    plan = phx.solver.SteadyVortexLatticePlan(
        surface,
        jnp.asarray((1.0, 0.0, 0.0)),
        wake_length=30.0,
        core_radius=0.02,
    )
    alpha = jnp.deg2rad(5.0)
    result = plan.solve(jnp.asarray((jnp.cos(alpha), 0.0, jnp.sin(alpha))))

    assert result.residual_norm < 1e-10
    assert result.total_force[2] > 0.0
    assert bool(result.successful)


def test_uvlm_sheds_only_on_accepted_step_and_fails_closed_on_capacity():
    surface = _wing(2)
    bound = phx.solver.SteadyVortexLatticePlan(
        surface,
        jnp.asarray((1.0, 0.0, 0.0)),
        wake_length=20.0,
        core_radius=0.02,
    )
    wake = phx.discretization.VortexWakePlan(
        surface.panel_count, surface.panel_count, 0.03
    )
    plan = phx.solver.UnsteadyVortexLatticePlan(bound, wake)
    initial = plan.initialize()
    first = plan.step(initial, jnp.asarray((1.0, 0.0, 0.1)), 0.01)
    second = plan.step(first.state, jnp.asarray((1.0, 0.0, 0.1)), 0.01)

    assert bool(first.successful)
    assert not bool(second.successful)
    np.testing.assert_allclose(
        second.state.wake.circulation, first.state.wake.circulation
    )


def test_cylinder_panel_solve_enforces_boundary_and_explicit_constraint():
    angle = jnp.linspace(0.0, 2.0 * jnp.pi, 25)
    vertices = jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
    geometry = phx.operators.FlowPanelGeometry2D.from_vertices(vertices)
    result = phx.solver.VortexPanelFlowPlan2D(
        geometry,
        prescribed_circulation=0.0,
    ).solve(jnp.asarray((1.0, 0.0)))

    assert result.boundary_residual_norm < 1e-10
    assert jnp.abs(result.constraint_residual) < 1e-10
    assert jnp.all(jnp.isfinite(result.pressure_coefficient))
    assert bool(result.successful)


def test_wall_transfer_and_bilinear_remesh_preserve_circulation_and_first_moment():
    angle = jnp.linspace(0.0, 2.0 * jnp.pi, 9)
    geometry = phx.operators.FlowPanelGeometry2D.from_vertices(
        jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
    )
    wall = phx.discretization.WallVorticityTransferPlan2D(16, 0.1, 0.15)
    transfer = wall.transfer(
        wall.initialize(dtype=float),
        geometry,
        jnp.linspace(-0.4, 0.4, geometry.length.size),
    )
    remesh = phx.discretization.ConservativeVortexRemeshPlan2D(
        (-2.0, -2.0),
        (2.0, 2.0),
        (16, 16),
        0.15,
    ).apply(
        transfer.accepted.position,
        transfer.accepted.circulation,
        transfer.accepted.active,
    )

    assert bool(transfer.successful)
    assert bool(remesh.successful)
    assert jnp.abs(remesh.circulation_residual) < 1e-12
    assert jnp.max(jnp.abs(remesh.first_moment_residual)) < 1e-12


def test_wall_transfer_overflow_preserves_the_accepted_pool():
    angle = jnp.linspace(0.0, 2.0 * jnp.pi, 9)
    geometry = phx.operators.FlowPanelGeometry2D.from_vertices(
        jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
    )
    plan = phx.discretization.WallVorticityTransferPlan2D(2, 0.1, 0.15)
    initial = plan.initialize(dtype=float)
    result = plan.transfer(
        initial,
        geometry,
        jnp.ones((geometry.length.size,)),
    )

    assert not bool(result.successful)
    assert int(result.overflow_count) == geometry.length.size - 2
    np.testing.assert_array_equal(result.accepted.active, initial.active)
    np.testing.assert_allclose(result.accepted.circulation, initial.circulation)
