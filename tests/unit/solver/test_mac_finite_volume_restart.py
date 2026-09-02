#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import ArrayArchiveCorruptionError
from phydrax.solver._mac_finite_volume_checkpoint import (
    MACFiniteVolumeCheckpointPlan,
    read_mac_finite_volume_checkpoint,
    write_mac_finite_volume_checkpoint,
)


def _adaptive_plan():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(4, periodic=True) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
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
    state = compiled.project_state(velocity)
    method = phx.solver.SSPRK33FixedStepMethod(compiled)
    controller = phx.solver.MACCompositeStepController(compiled)
    plan = phx.solver.MACAdaptiveRolloutPlan(
        compiled,
        method,
        controller,
        phx.solver.MACAdaptivePolicy(4, maximum_step_size=0.005),
        final_time=0.01,
        initial_step_size=0.005,
    )
    return plan, state


def test_mac_checkpoint_restore_continues_with_exact_controller_history(tmp_path):
    plan, state = _adaptive_plan()
    uninterrupted = plan.rollout(jnp.asarray(0.0), state)
    initial = plan.initialize(jnp.asarray(0.0), state)
    first = plan.advance(initial, jnp.asarray(0.005)).runtime_state
    checkpoint_plan = MACFiniteVolumeCheckpointPlan(plan, initial)
    path = tmp_path / "mac.chk"
    write_mac_finite_volume_checkpoint(path, checkpoint_plan, first)
    restored = read_mac_finite_volume_checkpoint(path, checkpoint_plan).runtime_state
    continued = plan.advance(restored, jnp.asarray(0.01)).runtime_state

    np.testing.assert_allclose(continued.state, uninterrupted.final_state)
    np.testing.assert_allclose(continued.time, 0.01)
    assert int(continued.accepted_step_count) == int(
        uninterrupted.grid.accepted_step_count
    )
    np.testing.assert_array_equal(
        continued.grid_valid_steps,
        uninterrupted.grid.valid_steps,
    )


def test_mac_checkpoint_corruption_fails_before_advance_and_preserves_runtime(
    tmp_path,
):
    plan, state = _adaptive_plan()
    runtime = plan.initialize(jnp.asarray(0.0), state)
    checkpoint_plan = MACFiniteVolumeCheckpointPlan(plan, runtime)
    path = tmp_path / "mac.chk"
    write_mac_finite_volume_checkpoint(path, checkpoint_plan, runtime)
    payload = path.read_bytes()
    path.write_bytes(payload[: len(payload) // 2])

    with pytest.raises(ArrayArchiveCorruptionError):
        read_mac_finite_volume_checkpoint(path, checkpoint_plan)
    np.testing.assert_array_equal(runtime.state, state)
    np.testing.assert_allclose(runtime.time, 0.0)
    assert int(runtime.accepted_step_count) == 0
