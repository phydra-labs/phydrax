"""Project one provider-native MuJoCo muscle through the MJX adapter."""

import jax
import jax.numpy as jnp
import mujoco

from phydrax.applications.robotics import prepare_mjx_adapter


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
complete_control = muscles.scatter_control(
    adapter.control().values.at[1].set(0.1), jnp.asarray([0.65])
)
stepped = adapter.step(adapter.initial_state, complete_control)
refreshed = adapter.refresh(stepped.accepted_state)
snapshot = muscles.snapshot(refreshed.accepted_state)

print("muscles", snapshot.names)
print("activation [1]", snapshot.activation.values)
print("compiled transmission length [m]", snapshot.length_m.values)
print("compiled transmission velocity [m/s]", snapshot.velocity_m_per_s.values)
print("raw provider force [N; negative pulls]", snapshot.raw_force_N.values)
print("fresh", bool(snapshot.freshness))
