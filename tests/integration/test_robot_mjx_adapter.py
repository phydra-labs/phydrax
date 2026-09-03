#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.applications.robotics._backend import RoboticsOperationStatus
from phydrax.applications.robotics._mjx import (
    mjx_availability,
    MJXObservationRequest,
    prepare_mjx_adapter,
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
    <motor name="motor" joint="hinge"/>
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
