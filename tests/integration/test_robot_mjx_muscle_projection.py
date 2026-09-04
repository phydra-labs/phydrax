#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.applications.robotics import (
    MJXMuscleProjectionPlan,
    prepare_mjx_adapter,
    RoboticsIndexEntry,
    RoboticsOperationStatus,
    RoboticsProjection,
    RoboticsProjectionMap,
)
from phydrax.dynamics import PlantStepContext


mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco.mjx")


_MUSCLE_MODEL = """
<mujoco>
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" solver="Newton"/>
  <worldbody>
    <site name="origin" pos="-0.45 0 0.15"/>
    <site name="wrap-side" pos="0 0.35 0.15"/>
    <geom name="wrap-sphere" type="sphere" pos="0 0 0.15" size="0.10" contype="0" conaffinity="0"/>
    <body name="shank" pos="0.45 0 0.15">
      <joint name="knee" type="hinge" axis="0 0 1" range="-0.7 0.7"/>
      <geom name="shank-mass" type="capsule" fromto="0 0 0 0.35 0 0" size="0.04" mass="1"/>
      <site name="insertion" pos="0.32 0 0"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="muscle-path">
      <site site="origin"/>
      <geom geom="wrap-sphere" sidesite="wrap-side"/>
      <site site="insertion"/>
    </spatial>
  </tendon>
  <actuator>
    <muscle name="soleus" tendon="muscle-path" force="1200"/>
    <motor name="assist" joint="knee"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def cpu_mjx_muscle():
    model = mujoco.MjModel.from_xml_string(_MUSCLE_MODEL)
    adapter = prepare_mjx_adapter(model, device=jax.devices("cpu")[0])
    return model, adapter, adapter.prepare_muscle_projection()


def _reset(adapter):
    return adapter.reset(
        jax.random.key(7),
        adapter.parameters,
    ).accepted_state


def _context(state):
    return PlantStepContext(
        state.time,
        state.time + jnp.asarray(0.001, dtype=state.time.dtype),
        state.step_index,
    )


def test_all_and_named_compiled_muscles_have_deterministic_fixed_maps(cpu_mjx_muscle):
    model, adapter, projection = cpu_mjx_muscle
    named = MJXMuscleProjectionPlan(("soleus",)).prepare(adapter)

    assert projection.names == ("soleus",)
    assert projection.names == named.names
    assert projection.prepared_id == named.prepared_id
    assert projection.actuator_indices == (0,)
    assert projection.activation_indices == (int(model.actuator_actadr[0]),)
    assert projection.activation_map.names == ("soleus",)
    assert projection.length_map.names == ("soleus",)
    assert projection.velocity_map.names == ("soleus",)
    assert projection.raw_force_map.names == ("soleus",)
    with pytest.raises(ValueError, match="not compiled built-in"):
        adapter.prepare_muscle_projection(("assist",))


def test_control_scatter_is_complete_and_snapshot_freshness_is_explicit(cpu_mjx_muscle):
    model, adapter, projection = cpu_mjx_muscle
    source = _reset(adapter)
    base = adapter.control(source)
    base = eqx.tree_at(
        lambda control: control.values,
        base,
        base.values.at[1].set(0.3),
    )
    complete = projection.scatter_control(base, jnp.asarray([0.7]))
    assert complete.values.shape == (model.nu,)
    assert complete.values[0] == pytest.approx(0.7)
    assert complete.values[1] == pytest.approx(0.3)

    initial = projection.snapshot(source)
    assert bool(initial.successful)
    assert bool(initial.freshness)
    assert initial.activation.values.shape == (1,)
    assert initial.length_m.values.shape == (1,)
    assert initial.velocity_m_per_s.values.shape == (1,)
    assert initial.raw_force_N.values.shape == (1,)
    assert initial.force_owner == "provider-native"
    assert initial.raw_force_sign == "negative-is-pulling-tension"

    stepped = adapter.step(
        _context(source),
        source,
        complete,
        adapter.parameters,
    )
    stale = projection.snapshot(stepped.accepted_state)
    assert int(stale.evidence.status) == int(RoboticsOperationStatus.INVALID_STATE)
    assert not bool(stale.freshness)
    refreshed = adapter.refresh(stepped.accepted_state)
    fresh = projection.snapshot(refreshed.accepted_state)
    assert bool(fresh.successful)
    assert bool(fresh.freshness)
    assert fresh.activation.values[0] > initial.activation.values[0]
    assert fresh.raw_force_N.values[0] <= 0.0


def test_scatter_rejects_wrong_kind_and_incomplete_typed_control_maps(
    cpu_mjx_muscle,
):
    _, adapter, projection = cpu_mjx_muscle
    source = _reset(adapter)
    control = adapter.control(source)
    excitation = jnp.asarray([0.5])

    wrong_kind_map = RoboticsProjectionMap(
        "qpos",
        adapter.control_map.size,
        adapter.control_map.entries,
        adapter.provenance,
    )
    wrong_kind = RoboticsProjection(control.values, wrong_kind_map)
    with pytest.raises(ValueError, match="kind 'control'"):
        projection.scatter_control(wrong_kind, excitation)

    different_map = RoboticsProjectionMap(
        "control",
        adapter.control_map.size,
        tuple(
            RoboticsIndexEntry(f"different-{index}", entry.start, entry.stop)
            for index, entry in enumerate(adapter.control_map.entries)
        ),
        adapter.provenance,
    )
    different = RoboticsProjection(control.values, different_map)
    with pytest.raises(ValueError, match="layout"):
        projection.scatter_control(different, excitation)

    partial_map = RoboticsProjectionMap(
        "control",
        1,
        (RoboticsIndexEntry("soleus", 0, 1),),
        adapter.provenance,
    )
    partial = RoboticsProjection(control.values[:1], partial_map)
    with pytest.raises(ValueError, match="layout"):
        projection.scatter_control(partial, excitation)


def test_failed_step_rolls_back_whole_state_then_refreshes_that_source(cpu_mjx_muscle):
    _, adapter, projection = cpu_mjx_muscle
    source = _reset(adapter)
    first_control = projection.scatter_control(
        adapter.control(source),
        jnp.asarray([0.6]),
    )
    first = adapter.step(
        _context(source),
        source,
        first_control,
        adapter.parameters,
    )
    assert bool(first.successful)
    assert not bool(projection.snapshot(first.accepted_state).freshness)

    invalid_control = adapter.control(first.accepted_state)
    invalid_control = eqx.tree_at(
        lambda control: control.values,
        invalid_control,
        invalid_control.values.at[0].set(jnp.nan),
    )
    failed = adapter.step(
        _context(first.accepted_state),
        first.accepted_state,
        invalid_control,
        adapter.parameters,
    )
    assert not bool(failed.successful)
    assert adapter.state_digest(failed.accepted_state) == adapter.state_digest(
        first.accepted_state
    )

    refreshed = adapter.refresh(failed.accepted_state)
    assert bool(refreshed.successful)
    assert bool(projection.snapshot(refreshed.accepted_state).freshness)


def test_scatter_and_forward_projection_support_jit_vmap_and_jvp(cpu_mjx_muscle):
    _, adapter, projection = cpu_mjx_muscle
    source = _reset(adapter)
    base = adapter.control(source).values
    bases = jnp.stack((base, base))
    excitations = jnp.asarray([[0.2], [0.8]])
    scattered = jax.jit(jax.vmap(projection.scatter_control))(bases, excitations)
    assert scattered.values.shape == (2, adapter.control_map.size)

    def raw_force_at_qpos(qpos):
        changed_data = source.payload.opaque.replace(qpos=qpos)
        changed_payload = eqx.tree_at(
            lambda payload: payload.opaque,
            source.payload,
            changed_data,
        )
        changed_state = eqx.tree_at(
            lambda state: state.payload,
            source,
            changed_payload,
        )
        refreshed = adapter.refresh(changed_state)
        return projection.snapshot(refreshed.accepted_state).raw_force_N.values

    _, tangent = jax.jvp(
        raw_force_at_qpos,
        (source.payload.opaque.qpos,),
        (jnp.ones_like(source.payload.opaque.qpos),),
    )
    assert tangent.shape == (1,)
    assert jnp.all(jnp.isfinite(tangent))
