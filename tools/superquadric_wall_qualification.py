#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualification campaign for superquadric triangle-wall contact."""

import contextlib
import io
import json
import runpy

import jax.numpy as jnp


output = io.StringIO()
with contextlib.redirect_stdout(output):
    namespace = runpy.run_path("examples/superquadric_triangle_wall.py")
result = namespace["result"]
response = result.evaluation.walls[0]
force_residual = jnp.linalg.norm(
    jnp.sum(response.particle_load.force, axis=0) + response.reaction_force
)
print(
    json.dumps(
        {
            "campaign": "superquadric-triangle-wall",
            "passed": bool(
                result.successful
                & (force_residual <= 1.0e-12)
                & (response.geometry.witness_residual[0] <= 1.0e-8)
            ),
            "gap": float(response.geometry.geometry.gap[0]),
            "feature_kind": int(response.geometry.feature_kind[0]),
            "witness_residual": float(response.geometry.witness_residual[0]),
            "feature_tie_margin": float(response.geometry.feature_tie_margin[0]),
            "force_residual": float(force_residual),
        },
        indent=2,
    )
)
