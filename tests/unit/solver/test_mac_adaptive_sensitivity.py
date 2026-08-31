#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _compiled(count=4):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    compiled = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 0.01),
        momentum,
        phx.solver.MACPressureProjectionPlan(operators, solve_method="transform"),
    )
    velocity = tuple(
        jnp.full(layout.shape, 0.1 * (axis + 1))
        for axis, layout in enumerate(finite_volume.face_layouts)
    )
    return compiled, compiled.project_state(velocity)


def test_adaptive_rollout_replay_and_terminal_derivatives_are_certified():
    compiled, state = _compiled()
    method = phx.solver.SSPRK33FixedStepMethod(compiled)
    controller = phx.solver.MACCompositeStepController(compiled, safety_factor=0.8)
    rollout = phx.solver.MACAdaptiveRolloutPlan(
        compiled,
        method,
        controller,
        phx.solver.MACAdaptivePolicy(4, maximum_step_size=0.005),
        final_time=0.01,
        initial_step_size=0.005,
    ).rollout(jnp.asarray(0.0), state)
    sensitivity = phx.solver.MACFixedGridSensitivityPlan(
        compiled,
        method,
        derivative_mode="smooth",
        checkpointing="full",
        block_size=None,
    )
    replay = sensitivity.rollout(state, rollout.grid)
    jvp = sensitivity.terminal_jvp(
        state,
        rollout.grid,
        None,
        initial_tangent=jnp.ones_like(state) * 1e-3,
    )
    vjp = sensitivity.terminal_vjp(state, rollout.grid, jnp.ones_like(state), None)

    assert rollout.successful
    assert replay.completed & replay.finite & replay.grid_valid
    assert jvp.successful
    assert vjp.successful
    assert jnp.all(jnp.isfinite(jvp.terminal_tangent))
    assert jnp.all(jnp.isfinite(vjp.initial_state_cotangent))


def test_segmented_shadowing_returns_explicit_certification_status():
    compiled, state = _compiled()
    method = phx.solver.SSPRK33FixedStepMethod(compiled)
    controller = phx.solver.MACCompositeStepController(compiled)
    rollout = phx.solver.MACAdaptiveRolloutPlan(
        compiled,
        method,
        controller,
        phx.solver.MACAdaptivePolicy(4, maximum_step_size=0.005),
        final_time=0.01,
        initial_step_size=0.005,
    ).rollout(jnp.asarray(0.0), state)
    sensitivity = phx.solver.MACFixedGridSensitivityPlan(
        compiled,
        method,
        derivative_mode="smooth",
        checkpointing="full",
        block_size=None,
    )
    shadowing = phx.solver.MACSegmentedShadowingPlan(
        sensitivity,
        segment_length=2,
        tangent_dimension=2,
        neutral_mode="flow",
        residual_tolerance=1e-4,
    ).solve(
        state,
        rollout.grid,
        None,
        None,
        lambda _time, coordinates, _args: 0.5 * jnp.vdot(coordinates, coordinates).real,
        observable_id="kinetic-coordinate-energy",
    )

    assert jnp.all(jnp.isfinite(shadowing.sensitivity)) | ~shadowing.successful
    assert shadowing.status >= 0
