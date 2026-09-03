"""Adapt, evaluate, and advance a native fixed-base hinge/prismatic robot."""

import json

import jax.numpy as jnp
import numpy as np

from phydrax.applications.robotics import parse_urdf_text
from phydrax.discretization import (
    reduced_forward_dynamics,
    reduced_inverse_dynamics,
    reduced_semi_implicit_velocity_euler_step,
)


URDF = """
<robot name="hinge-slider" version="1.0">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>
  <link name="hinge_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>
  <link name="slider_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="joint_0_hinge" type="revolute">
    <parent link="base"/>
    <child link="hinge_link"/>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.5" upper="2.5" effort="20" velocity="4"/>
    <dynamics damping="0.05"/>
  </joint>
  <joint name="joint_1_slider" type="prismatic">
    <parent link="hinge_link"/>
    <child link="slider_link"/>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.2" upper="0.3" effort="15" velocity="2"/>
    <dynamics damping="0.02"/>
  </joint>
</robot>
"""


adaptation = parse_urdf_text(URDF, root_policy="fixed_world")
if not adaptation.negotiation.valid or adaptation.evidence.loss_paths:
    raise RuntimeError("the self-contained URDF did not adapt losslessly")

particles = adaptation.particles.prepare()
bodies = adaptation.bodies.prepare(particles)
graph = adaptation.joints.prepare(bodies, adaptation.reference)
articulation = adaptation.articulation.prepare(graph, adaptation.reference)

if articulation.nq != 2 or articulation.nv != 2:
    raise RuntimeError("the hinge/prismatic tree must have two scalar DOFs")

expected_joint_ids = np.asarray(
    [
        adaptation.joint_ids.id_for_name("joint_0_hinge"),
        adaptation.joint_ids.id_for_name("joint_1_slider"),
    ],
    dtype=np.int64,
)
if not np.array_equal(np.asarray(articulation.joint_ids), expected_joint_ids):
    raise RuntimeError("prepared generalized coordinates have an unexpected order")

dtype = articulation.reference_position.dtype
configuration = jnp.asarray([0.35, 0.08], dtype=dtype)
velocity = jnp.asarray([0.20, -0.10], dtype=dtype)
kinematics = articulation.forward_kinematics(configuration, velocity)
if not bool(kinematics.successful):
    raise RuntimeError("forward kinematics rejected the finite state")

slider_id = int(adaptation.link_ids.id_for_name("slider_link"))
slider_transform = articulation.body_transform(configuration, slider_id)
angle = float(configuration[0])
extension = 0.5 + float(configuration[1])
expected_position = np.asarray(
    [
        0.5 + extension * np.cos(angle),
        extension * np.sin(angle),
        0.0,
    ]
)
if not np.allclose(
    np.asarray(slider_transform[:3, 3]),
    expected_position,
    rtol=1.0e-6,
    atol=1.0e-6,
):
    raise RuntimeError("forward kinematics disagrees with the analytic chain pose")

frame_jacobian = articulation.frame_jacobian_operator(configuration, slider_id)
frame_twist = frame_jacobian.mv(velocity)
expected_twist = np.asarray(
    [
        -extension * np.sin(angle) * float(velocity[0])
        + np.cos(angle) * float(velocity[1]),
        extension * np.cos(angle) * float(velocity[0])
        + np.sin(angle) * float(velocity[1]),
        0.0,
        0.0,
        0.0,
        float(velocity[0]),
    ]
)
if not np.allclose(
    np.asarray(frame_twist), expected_twist, rtol=1.0e-6, atol=1.0e-6
):
    raise RuntimeError("frame Jacobian action disagrees with analytic velocity")

acceleration = jnp.asarray([0.12, -0.07], dtype=dtype)
gravity = jnp.asarray([0.0, -9.81, 0.0], dtype=dtype)
inverse = reduced_inverse_dynamics(
    articulation,
    configuration,
    velocity,
    acceleration,
    gravity,
)
if not bool(inverse.successful):
    raise RuntimeError(f"inverse dynamics failed with status {int(inverse.status)}")

forward = reduced_forward_dynamics(
    articulation,
    configuration,
    velocity,
    inverse.generalized_effort,
    gravity,
)
if not bool(forward.successful):
    raise RuntimeError(f"forward dynamics failed with status {int(forward.status)}")
if not np.allclose(
    np.asarray(forward.acceleration),
    np.asarray(acceleration),
    rtol=2.0e-5,
    atol=2.0e-6,
):
    raise RuntimeError("inverse/forward dynamics reconstruction exceeded tolerance")

state = articulation.unpack_state(articulation.pack_state(configuration, velocity))
step = reduced_semi_implicit_velocity_euler_step(
    articulation,
    state,
    inverse.generalized_effort,
    gravity,
    jnp.asarray(1.0e-6, dtype=dtype),
)
if not bool(step.successful):
    raise RuntimeError(
        f"semi-implicit velocity Euler failed with status {int(step.status)}"
    )
if not np.allclose(
    np.asarray(step.accepted_state.configuration),
    np.asarray(step.candidate_state.configuration),
) or not np.allclose(
    np.asarray(step.accepted_state.velocity),
    np.asarray(step.candidate_state.velocity),
):
    raise RuntimeError("a successful step did not commit its candidate state")

print(
    json.dumps(
        {
            "checks_passed": True,
            "configuration": np.asarray(configuration).tolist(),
            "forward_inverse_relative_residual": float(
                forward.relative_inverse_forward_residual
            ),
            "frame_twist": np.asarray(frame_twist).tolist(),
            "generalized_effort": np.asarray(inverse.generalized_effort).tolist(),
            "recovered_acceleration": np.asarray(forward.acceleration).tolist(),
            "slider_position": np.asarray(slider_transform[:3, 3]).tolist(),
        },
        indent=2,
        sort_keys=True,
    )
)
