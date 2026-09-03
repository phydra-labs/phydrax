#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.applications.robotics import prepare_mjx_adapter


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
    <motor name="motor" joint="hinge"/>
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
