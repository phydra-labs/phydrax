#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualification campaign for conservative intraparticle AMR."""

import contextlib
import io
import json
import runpy

import jax.numpy as jnp


output = io.StringIO()
with contextlib.redirect_stdout(output):
    namespace = runpy.run_path("examples/adaptive_catalyst_pellet.py")
result = namespace["result"]
evidence = result.evidence
maximum_balance = max(
    float(jnp.max(jnp.abs(evidence.energy_residual))),
    float(jnp.max(jnp.abs(evidence.species_residual))),
    float(jnp.max(jnp.abs(evidence.pore_volume_residual))),
    float(jnp.max(jnp.abs(evidence.surface_area_residual))),
)
print(
    json.dumps(
        {
            "campaign": "particle-internal-amr",
            "passed": bool(result.successful & (maximum_balance <= 1.0e-12)),
            "selected_cells": int(evidence.selected_count),
            "active_fine_cells": int(jnp.sum(result.accepted_state.fine_active)),
            "maximum_balance_residual": maximum_balance,
            "route_digest": int(evidence.route_digest),
        },
        indent=2,
    )
)
