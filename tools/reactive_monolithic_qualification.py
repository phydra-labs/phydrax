#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualification campaign for monolithic fluid-particle Newton coupling."""

import contextlib
import io
import json
import runpy

import jax.numpy as jnp


output = io.StringIO()
with contextlib.redirect_stdout(output):
    namespace = runpy.run_path("examples/monolithic_reactive_cfd_dem.py")
result = namespace["result"]
evaluation = result.evaluation
print(
    json.dumps(
        {
            "campaign": "reactive-monolithic-newton",
            "passed": bool(
                result.successful
                & (jnp.linalg.norm(evaluation.momentum_residual) <= 1.0e-12)
                & (jnp.abs(evaluation.energy_residual) <= 1.0e-12)
                & (jnp.max(jnp.abs(evaluation.species_residual)) <= 1.0e-12)
            ),
            "nonlinear_status": int(result.nonlinear.status),
            "nonlinear_iterations": int(result.nonlinear.diagnostics.iterations),
            "momentum_residual": float(jnp.linalg.norm(evaluation.momentum_residual)),
            "energy_residual": float(evaluation.energy_residual),
            "species_residual": float(jnp.max(jnp.abs(evaluation.species_residual))),
            "preconditioner": result.preconditioner.mode.value,
        },
        indent=2,
    )
)
