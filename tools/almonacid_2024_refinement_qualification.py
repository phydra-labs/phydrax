#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Qualify short-horizon spatial and temporal convergence of the idealized MA continuum.

This is a native discretization study, not an external source or biological
validation. The external executable campaign is qualified separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import numpy as np

from phydrax.applications.skeletal_muscle.continuum import (
    almonacid_2024_repository_case,
)


ROOT = Path(__file__).resolve().parents[1]


def _rollout(inputs: Path, *, refinement: int, step_s: float, end_s: float) -> dict:
    count = round(end_s / step_s)
    if count < 1 or abs(count * step_s - end_s) > 1e-12:
        raise ValueError("end_s must be a positive integer multiple of step_s.")
    plan, parameters, history, _, source_end = almonacid_2024_repository_case(
        inputs, refinement=refinement
    )
    if end_s >= source_end:
        raise ValueError("The refinement horizon must precede the source endpoint.")
    prepared = plan.prepare(parameters)
    propose = eqx.filter_jit(lambda model, control: model.propose(control))
    maximum_residual = 0.0
    maximum_linear_iterations = 0
    for index in range(1, count + 1):
        candidate = propose(prepared, history.sample(index * step_s))
        jax.block_until_ready(candidate)
        if not bool(candidate.successful):
            raise ValueError(
                f"Refinement rollout failed at level {refinement}, step {index}: "
                f"nonlinear status {int(candidate.nonlinear_result.status)}."
            )
        diagnostics = candidate.nonlinear_result.diagnostics
        maximum_residual = max(maximum_residual, float(diagnostics.final_residual_norm))
        maximum_linear_iterations = max(
            maximum_linear_iterations, int(diagnostics.linear_iterations)
        )
        prepared = candidate.commit(prepared)
    final = candidate.diagnostics
    return {
        "refinement": refinement,
        "step_s": step_s,
        "steps": count,
        "plan_id": plan.plan_id,
        "reaction_pulling_x_N": float(final.reaction_pulling_N[0]),
        "current_volume_m3": float(final.current_volume_m3),
        "maximum_scaled_nonlinear_residual": maximum_residual,
        "maximum_linear_iterations": maximum_linear_iterations,
        "all_steps_successful": True,
    }


def _error(value: dict, reference: dict) -> float:
    actual = np.asarray((value["reaction_pulling_x_N"], value["current_volume_m3"]))
    target = np.asarray(
        (reference["reaction_pulling_x_N"], reference["current_volume_m3"])
    )
    scale = np.maximum(np.abs(target), np.asarray((1.0, 1e-6)))
    return float(np.linalg.norm((actual - target) / scale))


def _study(rows: list[dict]) -> dict:
    coarse_error = _error(rows[0], rows[2])
    intermediate_error = _error(rows[1], rows[2])
    floor = np.finfo(float).tiny
    return {
        "levels": rows,
        "coarse_to_fine_error": coarse_error,
        "intermediate_to_fine_error": intermediate_error,
        "observed_binary_refinement_order": float(
            np.log2(max(coarse_error, floor) / max(intermediate_error, floor))
        ),
        "passed": bool(
            np.isfinite(coarse_error)
            and np.isfinite(intermediate_error)
            and intermediate_error < coarse_error
        ),
    }


def qualify(inputs: Path, *, end_s: float) -> dict:
    spatial = [
        _rollout(inputs, refinement=level, step_s=0.01, end_s=end_s)
        for level in (0, 1, 2)
    ]
    temporal = [
        _rollout(inputs, refinement=0, step_s=step, end_s=end_s)
        for step in (0.02, 0.01, 0.005)
    ]
    spatial_study = _study(spatial)
    temporal_study = _study(temporal)
    return {
        "source_commit": "0698e3d87d7261c81437d410dda161fe3b8efcbf",
        "horizon_s": end_s,
        "spatial": spatial_study,
        "temporal": temporal_study,
        "passed": spatial_study["passed"] and temporal_study["passed"],
        "scope": (
            "Native short-horizon reaction/volume convergence; not external "
            "source equivalence, LBB certification, anatomy, or biology."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=ROOT / "tests/fixtures/flexodeal_0698e3d",
    )
    parser.add_argument("--end-time", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    result = qualify(args.inputs, end_s=args.end_time)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    args.output.write_text(payload + "\n")
    print(payload)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
