#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.applications.robotics import (
    mjx_availability,
    MJXObservationRequest,
    MJXRefreshResult,
    prepare_mjx_adapter,
    RoboticsIndexEntry,
    RoboticsOperationEvidence,
    RoboticsOperationStatus,
    RoboticsProjection,
    RoboticsProjectionMap,
)
from phydrax.backends._types import BackendUnavailableError


mujoco = pytest.importorskip("mujoco")
mjx = pytest.importorskip("mujoco.mjx")


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
  <sensor>
    <jointpos name="joint-angle" joint="hinge"/>
  </sensor>
</mujoco>
"""


def _model(*, mass="1", solver="Newton"):
    return mujoco.MjModel.from_xml_string(
        _MODEL.replace('mass="1"', f'mass="{mass}"').replace(
            '<option timestep="0.002"/>',
            f'<option timestep="0.002" solver="{solver}"/>',
        )
    )

def test_mjx_adapter_preserves_complete_state_and_refreshes_observations():
    model = _model()
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

def test_supported_provider_pair_reports_matching_qualified_release():
    availability = mjx_availability()

    assert availability.available
    versions = dict(availability.versions)
    assert versions["mujoco"].startswith("3.12.")
    assert versions["mujoco-mjx"].split(".post", 1)[0] == versions[
        "mujoco"
    ].split(".post", 1)[0]


def test_unsupported_model_feature_rejects_before_transfer(monkeypatch):
    transfers = []

    def transferred(*args, **kwargs):
        transfers.append((args, kwargs))
        raise AssertionError("unsupported model reached device transfer")

    monkeypatch.setattr(mjx, "put_model", transferred)

    with pytest.raises(BackendUnavailableError, match="unsupported solver"):
        prepare_mjx_adapter(_model(solver="PGS"))
    assert not transfers


def test_step_stales_sensors_and_explicit_refresh_returns_complete_fresh_state():
    model = _model()
    adapter = prepare_mjx_adapter(model)
    source = adapter.initial_state

    initial = adapter.observe(source)
    assert bool(initial.successful)
    assert bool(initial.projection.freshness)

    stepped = adapter.step(
        source, jnp.zeros((model.nu,), dtype=source.opaque.ctrl.dtype)
    )
    assert bool(stepped.successful)
    assert int(stepped.candidate_state.epoch) == 1
    assert int(stepped.accepted_state.epoch) == 1
    assert int(stepped.accepted_state.sensor_epoch) == 0

    stale = adapter.observe(stepped.accepted_state)
    assert int(stale.status) == int(RoboticsOperationStatus.INVALID_STATE)
    assert not bool(stale.successful)
    assert not bool(stale.projection.freshness)

    refreshed = adapter.refresh(stepped.accepted_state)
    assert bool(refreshed.successful)
    assert refreshed.candidate_state.opaque is not stepped.accepted_state.opaque
    assert int(refreshed.accepted_state.epoch) == 1
    assert int(refreshed.accepted_state.sensor_epoch) == 1
    assert bool(refreshed.observation.successful)
    assert bool(refreshed.observation.projection.freshness)
    assert adapter.qpos(refreshed.accepted_state).values.shape == (model.nq,)
    assert adapter.qvel(refreshed.accepted_state).values.shape == (model.nv,)
    assert adapter.control(refreshed.accepted_state).values.shape == (model.nu,)


def test_observation_request_selects_content_not_freshness():
    model = _model()
    adapter = prepare_mjx_adapter(model)
    request = MJXObservationRequest(
        qpos=False, qvel=True, control=False, sensors=False
    )

    observation = adapter.observe(adapter.initial_state, request)

    assert bool(observation.successful)
    assert observation.projection.index_map.names == ("qvel/hinge",)
    assert observation.projection.values.shape == (model.nv,)


def test_noncanonical_complete_data_leaf_rejects():
    adapter = prepare_mjx_adapter(_model())
    source = adapter.initial_state
    malformed_data = source.opaque.replace(
        qpos=jnp.concatenate((source.opaque.qpos, jnp.zeros((1,))))
    )
    malformed_state = eqx.tree_at(
        lambda state: state.opaque, source, malformed_data
    )

    with pytest.raises(ValueError, match="canonical intrinsic shape"):
        adapter.qpos(malformed_state)


def test_batched_nonfinite_candidate_rolls_back_only_its_case():
    model = _model()
    adapter = prepare_mjx_adapter(model)
    source = adapter.initial_state
    batched = jax.tree_util.tree_map(
        lambda leaf: jnp.stack((leaf, leaf)), source
    )
    controls = jnp.zeros((2, model.nu), dtype=source.opaque.ctrl.dtype)
    controls = controls.at[1, 0].set(jnp.nan)

    stepped = adapter.step(batched, controls)

    assert stepped.status.tolist() == [
        int(RoboticsOperationStatus.SUCCESS),
        int(RoboticsOperationStatus.NONFINITE),
    ]
    assert stepped.accepted_state.epoch.tolist() == [1, 0]
    assert stepped.accepted_state.sensor_epoch.tolist() == [0, 0]
    assert jnp.allclose(
        stepped.accepted_state.opaque.ctrl[0], controls[0]
    )
    assert jnp.allclose(
        stepped.accepted_state.opaque.ctrl[1], batched.opaque.ctrl[1]
    )
    observed = adapter.observe(stepped.accepted_state)
    assert observed.status.tolist() == [
        int(RoboticsOperationStatus.INVALID_STATE),
        int(RoboticsOperationStatus.SUCCESS),
    ]


def test_projection_and_state_provenance_mismatches_reject():
    first = prepare_mjx_adapter(_model(mass="1"))
    second = prepare_mjx_adapter(_model(mass="2"))

    assert first.provenance != second.provenance
    with pytest.raises(ValueError, match="different prepared adapter"):
        first.qpos(second.initial_state)
    with pytest.raises(ValueError, match="provenance"):
        first.step(first.initial_state, second.control())


def test_intrinsic_schema_records_every_make_data_array_leaf():
    adapter = prepare_mjx_adapter(_model())
    leaves = jax.tree_util.tree_leaves(adapter.initial_state.opaque)

    assert len(adapter.data_schema.leaves) == len(leaves)
    assert all(spec.initial_finite for spec in adapter.data_schema.leaves)
    assert all(spec.devices for spec in adapter.data_schema.leaves)
    assert adapter.data_schema.validate(adapter.initial_state.opaque) == ()
