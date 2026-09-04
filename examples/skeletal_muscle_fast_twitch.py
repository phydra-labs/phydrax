"""Run two source pulses through the Shorten 2007 fast-twitch cell."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np

from phydrax.applications.skeletal_muscle.cellular import (
    ShortenFastTwitchModel,
    ShortenIntegrationPlan,
)


def main() -> None:
    model = ShortenFastTwitchModel()
    time_grid_ms = np.linspace(0.0, 100.0, 201)
    trajectory = ShortenIntegrationPlan(model, time_grid_ms).prepare().integrate()
    ca2_index = model.state_layout.index("Ca_2")
    a2_index = model.state_layout.index("A_2")
    voltage_index = model.state_layout.index("vS")
    payload = {
        "model_id": model.model_id,
        "source_revision": model.source_revision,
        "all_steps_successful": bool(jnp.all(trajectory.successful)),
        "peak_sarcolemmal_voltage_mV": float(
            jnp.max(trajectory.states[:, voltage_index])
        ),
        "peak_bulk_cytosolic_calcium_uM": float(
            jnp.max(trajectory.states[:, ca2_index])
        ),
        "peak_force_bearing_crossbridge_uM": float(
            jnp.max(trajectory.states[:, a2_index])
        ),
        "force_owner": "Shorten razumova/A_2 biochemical tension driver",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
