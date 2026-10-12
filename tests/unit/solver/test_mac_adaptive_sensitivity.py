#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _compiled(count: Any = 4) -> Any:
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


def test_mac_adaptive_sensitivity_scenario_1() -> None:
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
    compiled, state = _compiled()
    plan = phx.solver.MACAdaptiveRolloutPlan(
        compiled,
        phx.solver.SSPRK33FixedStepMethod(compiled),
        phx.solver.MACCompositeStepController(compiled),
        phx.solver.MACAdaptivePolicy(2, maximum_step_size=0.005),
        final_time=0.02,
        initial_step_size=0.005,
    )
    runtime = plan.initialize(jnp.asarray(0.0), state)
    first = plan.advance(runtime, jnp.asarray(0.005)).runtime_state

    continued = plan.advance(first, jnp.asarray(0.02)).runtime_state

    assert continued.accepted_step_count == 2
    assert jnp.isclose(continued.time, 0.01)
    assert jnp.isclose(
        continued.time,
        continued.grid_times[continued.accepted_step_count],
    )
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


@pytest.mark.parametrize("regularization", [0.0, 0.2])
def test_mac_shadowing_dense_and_banded_nonzero_continuity(regularization: float) -> None:
    from phydrax.linalg._block_banded_least_squares import (
        solve_block_banded_least_squares,
    )
    from phydrax.solver._mac_sensitivity import _dense_shadowing_least_squares

    rng = np.random.default_rng(301)
    samples = jnp.asarray(rng.normal(size=(3, 6, 2)), dtype=jnp.float64)
    targets = jnp.asarray(rng.normal(size=(3, 6)), dtype=jnp.float64)
    factors = jnp.asarray(rng.normal(size=(3, 2, 2)), dtype=jnp.float64)
    offsets = jnp.asarray(rng.normal(size=(3, 2)), dtype=jnp.float64)
    dense, full_rank, minimum, condition = _dense_shadowing_least_squares(
        samples, targets, factors, offsets, 4.0, regularization, 1024 * 1024
    )
    banded = solve_block_banded_least_squares(
        samples,
        targets,
        -4.0 * factors[:-1],
        jnp.broadcast_to(4.0 * jnp.eye(2, dtype=jnp.float64), factors[:-1].shape),
        4.0 * offsets[:-1],
        regularization=regularization,
    )
    np.testing.assert_allclose(banded.value, dense, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(banded.minimum_singular_value, minimum, rtol=1e-7)
    np.testing.assert_allclose(banded.condition_number, condition, rtol=1e-6)
    assert full_rank & banded.successful


def test_mac_shadowing_dense_workspace_refuses_before_materialization() -> None:
    from phydrax.solver._mac_sensitivity import _dense_shadowing_least_squares

    with pytest.raises(ValueError, match="select block-banded"):
        _dense_shadowing_least_squares(
            jnp.ones((3, 6, 2), dtype=jnp.float64),
            jnp.ones((3, 6), dtype=jnp.float64),
            jnp.ones((3, 2, 2), dtype=jnp.float64),
            jnp.ones((3, 2), dtype=jnp.float64),
            4.0,
            0.0,
            1,
        )


@pytest.mark.parametrize("regularization", [0.0, 0.2])
def test_mac_shadowing_route_sensitivity_and_flow_neutral_parity(
    regularization: float,
) -> None:
    compiled, state = _compiled()
    method = phx.solver.SSPRK33FixedStepMethod(compiled)
    rollout = phx.solver.MACAdaptiveRolloutPlan(
        compiled,
        method,
        phx.solver.MACCompositeStepController(compiled),
        phx.solver.MACAdaptivePolicy(4, maximum_step_size=0.005),
        final_time=0.02,
        initial_step_size=0.005,
    ).rollout(jnp.asarray(0.0), state)
    sensitivity = phx.solver.MACFixedGridSensitivityPlan(
        compiled, method, derivative_mode="smooth", checkpointing="full", block_size=None
    )

    def observable(time: Any, coordinates: Any, parameters: Any) -> Any:
        del time
        return 0.5 * jnp.vdot(coordinates, coordinates).real + parameters

    dense = phx.solver.MACSegmentedShadowingPlan(
        sensitivity,
        segment_length=2,
        tangent_dimension=2,
        neutral_mode="flow",
        solve_route="dense-svd",
        regularization=regularization,
    ).solve(
        state,
        rollout.grid,
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observable,
        observable_id="route-parity-energy",
    )
    banded = phx.solver.MACSegmentedShadowingPlan(
        sensitivity,
        segment_length=2,
        tangent_dimension=2,
        neutral_mode="flow",
        regularization=regularization,
    ).solve(
        state,
        rollout.grid,
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observable,
        observable_id="route-parity-energy",
    )
    np.testing.assert_allclose(
        banded.segment_coefficients, dense.segment_coefficients, atol=1e-10
    )
    np.testing.assert_allclose(banded.shadow_tangent, dense.shadow_tangent, atol=1e-10)
    np.testing.assert_allclose(
        banded.response_samples, dense.response_samples, atol=1e-10
    )
    np.testing.assert_allclose(
        banded.sensitivity, dense.sensitivity, atol=1e-10, equal_nan=True
    )
    assert banded.status == dense.status
    assert banded.solve_route == "block-banded"
    assert dense.solve_route == "dense-svd"
