"""Project one provider-native MuJoCo muscle through the MJX adapter."""

import importlib

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax.applications.robotics import mjx_availability, prepare_mjx_adapter
from phydrax.dynamics import PlantStepContext


availability = mjx_availability()
if not availability.available:
    raise RuntimeError(
        f"MJX provider unavailable: {availability.reason} "
        f"(requires {availability.requirement})"
    )

mujoco = importlib.import_module("mujoco")


xml = """
<mujoco>
  <compiler autolimits="true"/>
  <option timestep="0.001" solver="Newton"/>
  <worldbody>
    <site name="origin" pos="-0.4 0 0.15"/><site name="side" pos="0 0.3 0.15"/>
    <geom name="wrap" type="sphere" pos="0 0 0.15" size="0.1" contype="0" conaffinity="0"/>
    <body pos="0.4 0 0.15">
      <joint name="hinge" type="hinge" range="-0.6 0.6"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" mass="1"/>
      <site name="insertion" pos="0.3 0 0"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="path">
      <site site="origin"/><geom geom="wrap" sidesite="side"/><site site="insertion"/>
    </spatial>
  </tendon>
  <actuator>
    <muscle name="soleus" tendon="path" force="1200"/>
    <motor name="assist" joint="hinge"/>
  </actuator>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
adapter = prepare_mjx_adapter(model, device=jax.devices("cpu")[0])
muscles = adapter.prepare_muscle_projection()
source = adapter.reset(jax.random.key(7), adapter.parameters).accepted_state
base_control = adapter.control(source)
base_control = eqx.tree_at(
    lambda control: control.values,
    base_control,
    base_control.values.at[1].set(0.1),
)
complete_control = muscles.scatter_control(base_control, jnp.asarray([0.65]))
context = PlantStepContext(
    source.time,
    source.time + jnp.asarray(model.opt.timestep, dtype=source.time.dtype),
    source.step_index,
)
stepped = adapter.step(
    context,
    source,
    complete_control,
    adapter.parameters,
)
if not bool(stepped.successful):
    raise RuntimeError(f"MJX step failed with status {int(stepped.status)}")
refreshed = adapter.refresh(stepped.accepted_state)
if not bool(refreshed.successful):
    raise RuntimeError(f"MJX refresh failed with status {int(refreshed.status)}")
snapshot = muscles.snapshot(refreshed.accepted_state)

print("muscles", snapshot.names)
print("activation [1]", snapshot.activation.values)
print("compiled transmission length [m]", snapshot.length_m.values)
print("compiled transmission velocity [m/s]", snapshot.velocity_m_per_s.values)
print("raw provider force [N; negative pulls]", snapshot.raw_force_N.values)
print("fresh", bool(snapshot.freshness))
