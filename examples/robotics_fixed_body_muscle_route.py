"""Evaluate a tensile route through two points fixed to articulated bodies."""

import jax.numpy as jnp

from phydrax.applications.robotics import FixedBodyRoutePlan, parse_urdf_text


urdf = """
<robot name="route-example">
  <link name="base">
    <inertial><mass value="1"/>
      <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="arm">
    <inertial><mass value="1"/>
      <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="elbow" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="1 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="200" velocity="10"/>
  </joint>
</robot>
"""
adaptation = parse_urdf_text(urdf)
particles = adaptation.particles.prepare()
bodies = adaptation.bodies.prepare(particles)
graph = adaptation.joints.prepare(bodies, adaptation.reference)
articulation = adaptation.articulation.prepare(graph, adaptation.reference)
base = int(adaptation.link_ids.id_for_name("base"))
arm = int(adaptation.link_ids.id_for_name("arm"))
route = FixedBodyRoutePlan(("elbow-flexor",), (0, 2), (base, arm)).prepare(
    articulation,
    jnp.asarray([[0.0, 0.25, 0.0], [0.35, -0.10, 0.0]]),
)
configuration = jnp.asarray([0.45])
velocity = jnp.asarray([-0.8])
evaluation = route.evaluate(configuration, velocity)
load, power = route.tensile_force_pullback(
    configuration, velocity, jnp.asarray([650.0])
)

print("route length [m]", evaluation.route_lengths_m)
print("route length rate [m/s]", evaluation.route_length_rates_m_per_s)
print("generalized tensile load [N m]", load)
print("virtual-power residual [W]", power.power_residual_W)
