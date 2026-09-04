#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.applications.robotics import (
    MJXRefreshResult,
    prepare_mjx_adapter,
    RoboticsIndexEntry,
    RoboticsOperationEvidence,
    RoboticsOperationStatus,
    RoboticsProjection,
    RoboticsProjectionMap,
)


mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco.mjx")


_MODEL = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="hinge" type="hinge"/>
      <geom name="ball" type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="hinge" joint="hinge"/>
  </actuator>
</mujoco>
"""


def test_mjx_adapter_preserves_complete_state_and_refreshes_observations():
    model = mujoco.MjModel.from_xml_string(_MODEL)
    adapter = prepare_mjx_adapter(model)

    source = adapter.initial_state
    result = adapter.step(
        source,
        jnp.zeros((model.nu,), dtype=source.opaque.ctrl.dtype),
        observations="both",
    )

    assert bool(result.evidence.successful)
    assert result.state.opaque is not source.opaque
    assert result.pre_step_observation is not None
    assert result.post_step_observation is not None
    assert result.pre_step_observation.freshness == "pre-step"
    assert result.post_step_observation.freshness == "post-step-refreshed"
    assert adapter.qpos(result.state).values.shape == (model.nq,)
    assert adapter.qvel(result.state).values.shape == (model.nv,)
    assert adapter.control(result.state).values.shape == (model.nu,)


def test_pre_step_observation_is_refreshed_after_control_installation():
    model = mujoco.MjModel.from_xml_string(_MODEL)
    adapter = prepare_mjx_adapter(model)
    control = jnp.asarray([0.625], dtype=adapter.initial_state.opaque.ctrl.dtype)

    result = adapter.step(
        adapter.initial_state,
        control,
        observations="pre",
    )

    assert bool(result.successful)
    assert result.pre_step_observation is not None
    entry = result.pre_step_observation.projection.index_map.entry("control/hinge")
    observed_control = result.pre_step_observation.projection.values[
        entry.start : entry.stop
    ]
    assert jnp.array_equal(observed_control, control)


@pytest.mark.parametrize(
    ("observation_mode", "failed_refresh_call"),
    (
        ("pre", 1),
        ("post", 1),
        ("both", 2),
    ),
)
def test_failed_refresh_overrides_successful_raw_observation_and_rolls_back_step(
    monkeypatch,
    observation_mode,
    failed_refresh_call,
):
    model = mujoco.MjModel.from_xml_string(_MODEL)
    adapter = prepare_mjx_adapter(model)
    original_refresh = type(adapter).refresh
    refresh_calls = 0

    def fail_requested_refresh(self, state=None, request=None, /):
        nonlocal refresh_calls
        refresh_calls += 1
        result = original_refresh(self, state, request)
        if refresh_calls != failed_refresh_call:
            return result
        assert bool(result.observation.successful)
        failure = RoboticsOperationEvidence(
            status=RoboticsOperationStatus.NONFINITE,
            finite=jnp.asarray(False),
            backend="mjx-jax",
            operation="sensors",
            implementation="fake.forward",
            device=self.device,
            dtype=self.dtype,
            detail="the requested fake provider refresh failed",
        )
        observation = result.observation
        return MJXRefreshResult(
            result.candidate_state,
            state,
            observation,
            failure,
            jnp.asarray(False),
        )

    monkeypatch.setattr(type(adapter), "refresh", fail_requested_refresh)
    control = jnp.asarray([0.75], dtype=adapter.initial_state.opaque.ctrl.dtype)
    result = adapter.step(
        adapter.initial_state,
        control,
        observations=observation_mode,
    )

    assert not bool(result.successful)
    assert not bool(result.evidence.finite)
    requested_observation = (
        result.pre_step_observation
        if observation_mode == "pre"
        else result.post_step_observation
    )
    assert requested_observation is not None
    assert not bool(requested_observation.successful)
    assert bool(result.rolled_back)
    assert jnp.array_equal(
        result.accepted_state.opaque.qpos,
        adapter.initial_state.opaque.qpos,
    )
    assert jnp.array_equal(
        result.accepted_state.opaque.ctrl,
        adapter.initial_state.opaque.ctrl,
    )
    assert int(result.accepted_state.epoch) == int(adapter.initial_state.epoch)


def test_step_rejects_wrong_kind_and_incomplete_typed_control_maps():
    model = mujoco.MjModel.from_xml_string(_MODEL)
    adapter = prepare_mjx_adapter(model)
    source = adapter.initial_state

    assert adapter.qpos_map.name_to_range == adapter.control_map.name_to_range
    assert adapter.qvel_map.name_to_range == adapter.control_map.name_to_range
    for wrong_kind in (adapter.qpos(source), adapter.qvel(source)):
        with pytest.raises(ValueError, match="kind 'control'"):
            adapter.step(source, wrong_kind)

    different_map = RoboticsProjectionMap(
        "control",
        adapter.control_map.size,
        (RoboticsIndexEntry("different", 0, adapter.control_map.size),),
        adapter.provenance,
    )
    different = RoboticsProjection(source.opaque.ctrl, different_map)
    with pytest.raises(ValueError, match="layout"):
        adapter.step(source, different)

    partial_map = RoboticsProjectionMap("control", 0, (), adapter.provenance)
    partial = RoboticsProjection(source.opaque.ctrl[:0], partial_map)
    with pytest.raises(ValueError, match="layout"):
        adapter.step(source, partial)
