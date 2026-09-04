"""Qualify MJX muscle projections against same-release host MuJoCo fields."""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.robotics import prepare_mjx_adapter


_MODEL = """
<mujoco>
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" solver="Newton"/>
  <worldbody>
    <site name="origin" pos="-0.4 0 0.15"/>
    <site name="side" pos="0 0.3 0.15"/>
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
    <muscle name="muscle" tendon="path" force="1000"/>
    <motor name="motor" joint="hinge"/>
  </actuator>
</mujoco>
"""


def qualify(steps: int) -> dict[str, object]:
    import mujoco

    model = mujoco.MjModel.from_xml_string(_MODEL)
    adapter = prepare_mjx_adapter(model, device=jax.devices("cpu")[0])
    projection = adapter.prepare_muscle_projection()
    state = adapter.initial_state
    control = projection.scatter_control(adapter.control(state), jnp.asarray([0.65]))
    for _ in range(steps):
        result = adapter.step(state, control)
        if not bool(result.successful):
            raise RuntimeError("MJX candidate failed before provider comparison.")
        state = result.accepted_state
    host = mujoco.MjData(model)
    host.qpos[:] = np.asarray(state.opaque.qpos)
    host.qvel[:] = np.asarray(state.opaque.qvel)
    host.act[:] = np.asarray(state.opaque.act)
    host.ctrl[:] = np.asarray(state.opaque.ctrl)
    host.time = float(np.asarray(state.opaque.time))
    mujoco.mj_forward(model, host)
    refreshed = adapter.refresh(state)
    if not bool(refreshed.successful):
        raise RuntimeError("MJX forward refresh failed before provider comparison.")
    snapshot = projection.snapshot(refreshed.accepted_state)
    actuator_index = projection.actuator_indices[0]
    activation_index = projection.activation_indices[0]
    host_values = np.asarray(
        [
            host.act[activation_index],
            host.actuator_length[actuator_index],
            host.actuator_velocity[actuator_index],
            host.actuator_force[actuator_index],
        ]
    )
    mjx_values = np.asarray(
        [
            snapshot.activation.values[0],
            snapshot.length_m.values[0],
            snapshot.velocity_m_per_s.values[0],
            snapshot.raw_force_N.values[0],
        ]
    )
    errors = np.abs(host_values - mjx_values)
    tolerance = 2.0e-4
    return {
        "provider": adapter.provenance.provider,
        "comparison": "same-state-forward-projection",
        "state_generation_steps": steps,
        "step_parity_evaluated": False,
        "fields": [
            "activation",
            "actuator_length_m",
            "actuator_velocity_m_per_s",
            "raw_provider_force_N",
        ],
        "host_values": host_values.tolist(),
        "mjx_values": mjx_values.tolist(),
        "absolute_errors": errors.tolist(),
        "tolerance": tolerance,
        "fresh": bool(snapshot.freshness),
        "passed": bool(snapshot.successful) and float(np.max(errors)) <= tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.steps <= 0:
        raise ValueError("--steps must be positive.")
    report = qualify(arguments.steps)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
