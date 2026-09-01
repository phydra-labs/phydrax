#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import contextlib
import io
import json
import runpy

import jax.numpy as jnp


def _run(path):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        return runpy.run_path(path)


pair = _run("examples/vortex_particle_pair.py")
periodic = _run("examples/vortex_periodic_vic.py")
viscous = _run("examples/vortex_pse_diffusion.py")
three_dimensional = _run("examples/vortex_3d_stretching.py")
lifting = _run("examples/vortex_lifting_surface.py")
panels = _run("examples/vortex_panel_cylinder.py")

pair_diagnostics = pair["diagnostics"]
periodic_backend = periodic["backend"]
diffusion = viscous["diffusion_evaluation"]
three_diagnostics = three_dimensional["diagnostics"]
lifting_result = lifting["result"]
unsteady = lifting["unsteady_result"]
panel_result = panels["result"]
wall_result = panels["wall_result"]

metrics = {
    "pair_circulation_defect": float(jnp.abs(pair_diagnostics.total_strength - 2.0)),
    "pair_impulse_norm": float(jnp.linalg.norm(pair_diagnostics.linear_impulse)),
    "periodic_compatibility_residual": float(periodic_backend.compatibility_residual),
    "periodic_balance_defect": float(periodic_backend.balance_defect),
    "periodic_divergence_norm": float(periodic_backend.divergence_norm),
    "pse_total_rate_defect": float(jnp.max(jnp.abs(diffusion.diagnostics.total_rate))),
    "three_dimensional_finite": bool(three_diagnostics.finite),
    "lifting_residual": float(lifting_result.residual_norm),
    "uvlm_circulation_residual": float(unsteady.wake_conservation_residual),
    "panel_boundary_residual": float(panel_result.boundary_residual_norm),
    "panel_constraint_residual": float(jnp.abs(panel_result.constraint_residual)),
    "wall_transfer_residual": float(jnp.abs(wall_result.circulation_residual)),
}
passed = bool(
    metrics["pair_circulation_defect"] < 1.0e-12
    and metrics["pair_impulse_norm"] < 1.0e-12
    and metrics["periodic_compatibility_residual"] < 1.0e-12
    and metrics["periodic_balance_defect"] < 1.0e-12
    and metrics["periodic_divergence_norm"] < 1.0e-10
    and metrics["pse_total_rate_defect"] < 1.0e-12
    and metrics["three_dimensional_finite"]
    and metrics["lifting_residual"] < 1.0e-10
    and metrics["uvlm_circulation_residual"] < 1.0e-12
    and metrics["panel_boundary_residual"] < 1.0e-10
    and metrics["panel_constraint_residual"] < 1.0e-10
    and metrics["wall_transfer_residual"] < 1.0e-12
)
print(json.dumps({"campaign": "vortex-methods", "passed": passed, **metrics}, indent=2))
if not passed:
    raise SystemExit(1)
