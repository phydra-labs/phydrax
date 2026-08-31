#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualification campaign for transactional particle-capacity growth."""

import contextlib
import io
import json
import runpy


output = io.StringIO()
with contextlib.redirect_stdout(output):
    namespace = runpy.run_path("examples/growing_reactive_particle_pool.py")
result = namespace["result"]
passed = bool(
    result.successful
    & (result.epoch.dynamics.bodies.capacity > namespace["particles"].capacity)
    & (abs(result.transition.mass_residual) <= 1.0e-12)
)
print(
    json.dumps(
        {
            "campaign": "particle-capacity-growth",
            "passed": passed,
            "old_capacity": namespace["particles"].capacity,
            "new_capacity": result.epoch.dynamics.bodies.capacity,
            "mass_residual": float(result.transition.mass_residual),
            "momentum_residual": [
                float(value) for value in result.transition.momentum_residual
            ],
        },
        indent=2,
    )
)
