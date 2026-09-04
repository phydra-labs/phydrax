#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax._array_tree import ArrayPyTreeSchema
from phydrax._identity import ExecutableSignature, NumericRevision, SemanticProvenance
from phydrax.applications.robotics import (
    mjx_availability,
    MJXObservationRequest,
    MJXState,
    prepare_mjx_adapter,
    RoboticsOperationStatus,
)
from phydrax.applications.robotics._backend import RoboticsOperationRequirement
from phydrax.backends._types import BackendUnavailableError
from phydrax.dynamics._plant import (
    AbstractDiscretePlant,
    PlantRuntimeState,
    PlantStepContext,
)


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


def _reset(adapter, case_shape=()):
    if case_shape:
        keys = jax.random.split(jax.random.key(7), case_shape[0])
    else:
        keys = jax.random.key(7)
    return adapter.reset(
        keys,
        adapter.parameters,
        case_shape=case_shape,
        initial_time=jnp.zeros(case_shape, dtype=jnp.float32),
    )


def _context(state):
    return PlantStepContext(
        state.time,
        state.time + jnp.asarray(0.002, dtype=state.time.dtype),
        state.step_index,
    )


def test_supported_provider_pair_reports_matching_qualified_release():
    availability = mjx_availability()

    assert availability.available
    versions = dict(availability.versions)
    assert versions["mujoco"].startswith("3.12.")
    assert (
        versions["mujoco-mjx"].split(".post", 1)[0]
        == versions["mujoco"].split(".post", 1)[0]
    )


def test_prepared_plant_keeps_closed_manifest_device_and_dtype():
    plant = prepare_mjx_adapter(_model())
    other_device = "gpu" if plant.device != "gpu" else "cpu"
    other_dtype = "float64" if plant.dtype != "float64" else "float32"

    assert isinstance(plant, AbstractDiscretePlant)
    assert plant.feature_manifest.solver == "newton"
    assert isinstance(plant.semantic_provenance, SemanticProvenance)
    assert isinstance(plant.numeric_revision, NumericRevision)
    assert isinstance(plant.execution_signature, ExecutableSignature)
    for operation in ("step", "sensors"):
        capability = plant.profile.capability(operation)
        assert capability.devices == (plant.device,)
        assert capability.dtypes == (plant.dtype,)
    step_capability = plant.profile.capability("step")
    assert step_capability.solvers == (plant.feature_manifest.solver,)
    assert step_capability.contact_features == plant.feature_manifest.contact_features
    plant.profile.require(
        (RoboticsOperationRequirement("step", device=plant.device, dtype=plant.dtype),)
    )
    with pytest.raises(BackendUnavailableError, match="device"):
        plant.profile.require(
            (RoboticsOperationRequirement("step", device=other_device),)
        )
    with pytest.raises(BackendUnavailableError, match="dtype"):
        plant.profile.require((RoboticsOperationRequirement("step", dtype=other_dtype),))


def test_unsupported_model_feature_rejects_before_transfer(monkeypatch):
    transfers = []

    def transferred(*args, **kwargs):
        transfers.append((args, kwargs))
        raise AssertionError("unsupported model reached device transfer")

    monkeypatch.setattr(mjx, "put_model", transferred)

    with pytest.raises(BackendUnavailableError, match="unsupported solver"):
        prepare_mjx_adapter(_model(solver="PGS"))
    assert not transfers


def test_reset_step_refresh_and_observe_use_plant_runtime_state():
    model = _model()
    plant = prepare_mjx_adapter(model)

    reset = _reset(plant)
    assert isinstance(reset.accepted_state, PlantRuntimeState)
    assert isinstance(reset.accepted_state.payload, MJXState)
    assert bool(reset.attempted)
    assert bool(reset.successful)
    assert int(reset.status) == 0
    source = reset.accepted_state

    initial = plant.observe(source)
    assert bool(initial.successful)
    assert bool(initial.projection.freshness)
    command = plant.control(source)
    assert bool(command.freshness)

    stepped = plant.step(_context(source), source, command, plant.parameters)
    assert bool(stepped.attempted)
    assert bool(stepped.successful)
    assert int(stepped.candidate_state.payload.epoch) == 1
    assert int(stepped.accepted_state.payload.epoch) == 1
    assert int(stepped.accepted_state.payload.sensor_epoch) == 0
    assert int(stepped.accepted_state.step_index) == 1

    stale = plant.observe(stepped.accepted_state)
    assert int(stale.status) == int(RoboticsOperationStatus.INVALID_STATE)
    assert not bool(stale.successful)
    assert not bool(stale.projection.freshness)

    refreshed = plant.refresh(stepped.accepted_state)
    assert bool(refreshed.attempted)
    assert bool(refreshed.successful)
    assert isinstance(refreshed.accepted_state, PlantRuntimeState)
    assert int(refreshed.accepted_state.payload.epoch) == 1
    assert int(refreshed.accepted_state.payload.sensor_epoch) == 1
    assert bool(refreshed.observation.successful)
    assert bool(refreshed.observation.projection.freshness)
    assert plant.qpos(refreshed.accepted_state).values.shape == (model.nq,)
    assert plant.qvel(refreshed.accepted_state).values.shape == (model.nv,)
    assert plant.control(refreshed.accepted_state).values.shape == (model.nu,)


def test_observation_request_selects_content_not_freshness():
    model = _model()
    plant = prepare_mjx_adapter(model)
    source = _reset(plant).accepted_state
    request = MJXObservationRequest(qpos=False, qvel=True, control=False, sensors=False)

    observation = plant.observe(source, request)

    assert bool(observation.successful)
    assert observation.projection.index_map.names == ("qvel/hinge",)
    assert observation.projection.values.shape == (model.nv,)


def test_shared_schema_covers_complete_payload_and_exact_case_axes():
    plant = prepare_mjx_adapter(_model(), case_ndim=1)
    state = _reset(plant, (2,)).accepted_state
    provider_leaves = jax.tree_util.tree_leaves(state.payload.opaque)
    payload_leaves = jax.tree_util.tree_leaves(state.payload)

    assert isinstance(plant.state_schema, ArrayPyTreeSchema)
    assert isinstance(plant.control_schema, ArrayPyTreeSchema)
    assert plant.state_schema.case_ndim == 1
    assert len(plant.state_schema.leaves) == len(provider_leaves) + 2
    assert plant.state_schema.validate(state.payload) == (2,)
    assert plant.control_schema.validate(plant.control(state)) == (2,)
    assert plant.parameter_schema.validate(plant.parameters.values) == ()
    assert all(leaf.shape[0] == 2 for leaf in payload_leaves)

    malformed_data = state.payload.opaque.replace(
        qpos=jnp.concatenate((state.payload.opaque.qpos, jnp.zeros((2, 1))), axis=-1)
    )
    malformed_payload = eqx.tree_at(
        lambda payload: payload.opaque, state.payload, malformed_data
    )
    malformed = eqx.tree_at(lambda runtime: runtime.payload, state, malformed_payload)
    with pytest.raises(ValueError, match="intrinsic shape"):
        plant.qpos(malformed)


def test_stale_and_wrong_control_projection_are_rejected():
    first = prepare_mjx_adapter(_model(mass="1"))
    second = prepare_mjx_adapter(_model(mass="2"))
    first_state = _reset(first).accepted_state
    second_state = _reset(second).accepted_state
    command = first.control(first_state)

    stepped = first.step(_context(first_state), first_state, command, first.parameters)
    stale = first.step(
        _context(stepped.accepted_state),
        stepped.accepted_state,
        command,
        first.parameters,
    )
    assert not bool(stale.attempted)
    assert not bool(stale.successful)
    assert int(stale.status) == int(RoboticsOperationStatus.INVALID_STATE)
    assert int(stale.candidate_state.payload.epoch) == 2
    assert int(stale.accepted_state.payload.epoch) == 1
    assert first.state_digest(stale.accepted_state) == first.state_digest(
        stepped.accepted_state
    )

    with pytest.raises(ValueError, match="PyTree"):
        first.step(
            _context(first_state),
            first_state,
            second.control(second_state),
            first.parameters,
        )
    for wrong_kind in (first.qpos(first_state), first.qvel(first_state)):
        with pytest.raises(ValueError, match="PyTree"):
            first.step(_context(first_state), first_state, wrong_kind, first.parameters)


def test_one_nonfinite_case_rolls_back_the_complete_payload_only_for_that_case():
    plant = prepare_mjx_adapter(_model(), case_ndim=1)
    source = _reset(plant, (2,)).accepted_state
    command = plant.control(source)
    command = eqx.tree_at(
        lambda projection: projection.values,
        command,
        command.values.at[1, 0].set(jnp.nan),
    )

    stepped = plant.step(_context(source), source, command, plant.parameters)

    assert stepped.attempted.tolist() == [True, True]
    assert stepped.successful.tolist() == [True, False]
    assert stepped.status.tolist() == [
        int(RoboticsOperationStatus.SUCCESS),
        int(RoboticsOperationStatus.NONFINITE),
    ]
    assert stepped.candidate_state.payload.epoch.tolist() == [1, 1]
    assert stepped.accepted_state.payload.epoch.tolist() == [1, 0]
    assert stepped.accepted_state.payload.sensor_epoch.tolist() == [0, 0]
    for accepted_leaf, source_leaf in zip(
        jax.tree_util.tree_leaves(stepped.accepted_state.payload),
        jax.tree_util.tree_leaves(source.payload),
        strict=True,
    ):
        assert jnp.array_equal(accepted_leaf[1], source_leaf[1])


def test_refresh_rolls_back_only_the_nonfinite_complete_case(monkeypatch):
    plant = prepare_mjx_adapter(_model(), case_ndim=1)
    source = _reset(plant, (2,)).accepted_state
    marked_data = source.payload.opaque.replace(
        qpos=source.payload.opaque.qpos.at[1, 0].set(1.0)
    )
    marked_payload = eqx.tree_at(
        lambda payload: payload.opaque, source.payload, marked_data
    )
    source = eqx.tree_at(lambda runtime: runtime.payload, source, marked_payload)
    provider_forward = mjx.forward

    def forward_with_one_bad_case(model, data):
        forwarded = provider_forward(model, data)
        failed = data.qpos[0] > 0.5
        return forwarded.replace(
            qpos=jnp.where(
                failed,
                jnp.full_like(forwarded.qpos, jnp.nan),
                forwarded.qpos,
            )
        )

    monkeypatch.setattr(mjx, "forward", forward_with_one_bad_case)

    refreshed = plant.refresh(source)

    assert refreshed.attempted.tolist() == [True, True]
    assert refreshed.successful.tolist() == [True, False]
    assert refreshed.status.tolist() == [
        int(RoboticsOperationStatus.SUCCESS),
        int(RoboticsOperationStatus.NONFINITE),
    ]
    assert jnp.isnan(refreshed.candidate_state.payload.opaque.qpos[1, 0])
    for accepted_leaf, source_leaf in zip(
        jax.tree_util.tree_leaves(refreshed.accepted_state.payload),
        jax.tree_util.tree_leaves(source.payload),
        strict=True,
    ):
        assert jnp.array_equal(accepted_leaf[1], source_leaf[1])


def test_generic_checkpoint_and_replay_digest_without_serializing_provider_objects():
    plant = prepare_mjx_adapter(_model())
    source = _reset(plant).accepted_state
    checkpoint = plant.checkpoint(source)
    context = _context(source)
    command = plant.control(source)
    expected = plant.step(context, source, command, plant.parameters)
    expected_digest = plant.state_digest(expected.accepted_state)

    replay = plant.replay(
        checkpoint,
        (context,),
        (command,),
        plant.parameters,
        expected_digests=(expected_digest,),
    )

    assert plant.verify_checkpoint(checkpoint)
    restored = plant.restore(checkpoint)
    assert isinstance(restored.payload.opaque, mjx.Data)
    assert plant.state_digest(restored) == checkpoint.digest
    assert replay.matched
    assert replay.first_mismatch_step == -1
    assert plant.state_digest(replay.final_state) == expected_digest
    assert all(
        not isinstance(leaf, (mujoco.MjModel, mjx.Model))
        for leaf in jax.tree_util.tree_leaves(checkpoint)
    )
