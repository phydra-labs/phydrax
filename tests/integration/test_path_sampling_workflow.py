#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.stochastic.path_sampling import (
    DeterministicPathAction,
    DynamicsKernelCapabilities,
    DynamicsStep,
    FunctionalDynamicsKernel,
    initialize_retis,
    initialize_tis,
    initialize_tps,
    InterfaceNetworkPlan,
    PATH_PROPAGATION_KERNEL_FAILURE,
    PathBuffer,
    prepare_retis,
    prepare_tis,
    prepare_tps,
    read_retis_restart,
    read_tps_restart,
    retis_step,
    RETISPlan,
    StateRegionPlan,
    tis_step,
    TISPlan,
    tps_step,
    TPSPlan,
    write_retis_restart,
    write_tps_restart,
)


def _kernel(*, fail: bool = False) -> FunctionalDynamicsKernel:
    def transition(source, destination, direction):
        del direction
        displacement = jnp.abs(destination - source)
        return jnp.where(jnp.all(jnp.abs(displacement - 1.0) < 1.0e-6), 0.0, -jnp.inf)

    def step(key, state, direction):
        del key
        valid = jnp.asarray(not fail)
        return DynamicsStep(
            state + direction.astype(state.dtype),
            jnp.asarray(0.0),
            valid,
            jnp.where(valid, 0, PATH_PROPAGATION_KERNEL_FAILURE),
        )

    return FunctionalDynamicsKernel(
        step,
        transition,
        DynamicsKernelCapabilities(
            stochastic=False,
            reversible=True,
            supports_backward=True,
            normalized_transition_density=False,
        ),
        time_step=1.0,
        kernel_id="workflow-failing" if fail else "workflow-deterministic",
    )


def _reactive_path() -> PathBuffer:
    return PathBuffer.from_trajectory(
        jnp.arange(5.0).reshape((5, 1)),
        jnp.arange(5.0),
        capacity=8,
    )


def _network() -> InterfaceNetworkPlan:
    basin_a = StateRegionPlan.half_open(jnp.asarray([-0.5]), jnp.asarray([0.5]))
    basin_b = StateRegionPlan.half_open(jnp.asarray([3.5]), jnp.asarray([4.5]))
    return InterfaceNetworkPlan(
        basin_a,
        basin_b,
        lambda states: states[..., 0],
        (0.5, 1.5),
        coordinate_id="workflow-coordinate",
    )


def test_tps_rejected_move_restart_preserves_exact_lineage(tmp_path) -> None:
    kernel = _kernel(fail=True)
    prepared = prepare_tps(
        TPSPlan(
            _network().ensemble(0),
            kernel,
            DeterministicPathAction(kernel),
            move_kind="one-way-shooting",
            lineage_capacity=16,
        ),
        _reactive_path(),
    )
    initial = initialize_tps(prepared)
    result = tps_step(prepared, initial, jax.random.key(10))
    assert not bool(result.move.accepted)
    assert int(result.state.rejected_count) == 1
    assert not bool(result.state.lineage.accepted[0])
    assert int(result.state.lineage.parent[0]) == int(result.state.lineage.committed[0])

    destination = tmp_path / "tps-restart.pxa"
    write_tps_restart(destination, prepared, result.state)
    restored = read_tps_restart(destination, prepared)
    np.testing.assert_array_equal(
        restored.state.path.positions, result.state.path.positions
    )
    np.testing.assert_array_equal(restored.state.path.lineage, result.state.path.lineage)
    np.testing.assert_array_equal(
        restored.state.lineage.parent, result.state.lineage.parent
    )
    np.testing.assert_array_equal(
        restored.state.lineage.candidate, result.state.lineage.candidate
    )
    np.testing.assert_array_equal(
        restored.state.lineage.committed, result.state.lineage.committed
    )
    np.testing.assert_array_equal(
        restored.state.lineage.accepted, result.state.lineage.accepted
    )
    assert restored.prepared_id == prepared.prepared_id
    assert restored.plan_id == prepared.plan.plan_id


def test_tis_and_retis_prepared_workflows_preserve_interface_assignments(
    tmp_path,
) -> None:
    kernel = _kernel()
    action = DeterministicPathAction(kernel)
    plan = TISPlan(_network(), kernel, action, move_kind="two-way-shooting")
    prepared = prepare_tis(plan, (_reactive_path(), _reactive_path()))
    state = initialize_tis(prepared)
    stepped = tis_step(prepared, state, jax.random.key(20), replica_index=1)
    assert int(stepped.state.step_index) == 1
    assert bool(plan.network.ensemble(1).contains(stepped.state.replicas[1].path))

    minus_path = PathBuffer.from_trajectory(
        jnp.asarray([[0.0], [1.0], [2.0], [1.0], [0.0]]),
        jnp.arange(5.0),
        capacity=8,
    )
    retis_prepared = prepare_retis(
        RETISPlan(plan),
        (minus_path, _reactive_path(), _reactive_path()),
    )
    retis_state = initialize_retis(retis_prepared)
    exchange = retis_step(
        retis_prepared,
        retis_state,
        jax.random.key(21),
        move_kind="exchange",
        replica_index=1,
    )
    assert bool(exchange.accepted)
    np.testing.assert_allclose(exchange.evaluation.exchange_log_ratio, 0.0)
    assert int(exchange.state.exchange_count) == 1
    assert int(exchange.state.accepted_exchange_count) == 1
    assert int(exchange.state.replicas[1].trajectory_serial) != int(
        exchange.state.replicas[2].trajectory_serial
    )
    shooting = retis_step(
        retis_prepared,
        exchange.state,
        jax.random.key(22),
        move_kind="shooting",
        replica_index=1,
    )
    lineage = shooting.state.replicas[1].lineage
    assert int(lineage.candidate[1]) != int(lineage.parent[1])

    restart_path = tmp_path / "retis-restart.pxa"
    write_retis_restart(restart_path, retis_prepared, exchange.state)
    restored = read_retis_restart(restart_path, retis_prepared)
    np.testing.assert_array_equal(
        restored.state.replicas[1].lineage.accepted,
        exchange.state.replicas[1].lineage.accepted,
    )
    assert int(restored.state.accepted_exchange_count) == 1


def test_tps_step_has_a_fixed_shape_compiled_runtime() -> None:
    kernel = _kernel()
    prepared = prepare_tps(
        TPSPlan(
            _network().ensemble(0),
            kernel,
            DeterministicPathAction(kernel),
            move_kind="two-way-shooting",
        ),
        _reactive_path(),
    )
    state = initialize_tps(prepared)
    execute = jax.jit(lambda current, key: tps_step(prepared, current, key).state)
    result = execute(state, jax.random.key(30))
    assert result.path.positions.shape == state.path.positions.shape
    assert result.lineage.parent.shape == state.lineage.parent.shape
    assert int(result.step_index) == 1
    assert (
        result.last_evaluation.log_acceptance_ratio.dtype
        == state.last_evaluation.log_acceptance_ratio.dtype
    )
