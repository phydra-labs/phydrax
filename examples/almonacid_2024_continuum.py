# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Run the idealized source muscle/aponeurosis with atomic state acceptance.

Repository inputs are an executable reference, not a biological preset. The
example defaults to a coarser mesh; use --refinement 2 for source mesh identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax

from phydrax.applications.skeletal_muscle.continuum import almonacid_2024_repository_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/flexodeal_0698e3d",
    )
    parser.add_argument("--refinement", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--protocol",
        choices=("repository-default-fields", "fixed-end-activation", "passive-cyclic"),
        default="repository-default-fields",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    jax.config.update("jax_enable_x64", True)
    plan, parameters, history, dt, end = almonacid_2024_repository_case(
        args.inputs, refinement=args.refinement, protocol=args.protocol
    )
    if args.steps * dt >= end:
        parser.error("--steps exceeds the reference input interval")
    prepared = plan.prepare(parameters)
    propose = eqx.filter_jit(lambda model, control: model.propose(control))
    for step in range(1, args.steps + 1):
        candidate = propose(prepared, history.sample(step * dt))
        jax.block_until_ready(candidate)
        prepared = candidate.commit(prepared)
        diagnostics = candidate.diagnostics
        print(
            json.dumps(
                {
                    "step": step,
                    "time_s": float(prepared.state.time_s),
                    "committed": bool(candidate.successful),
                    "nonlinear_status": int(candidate.nonlinear_result.status),
                    "initial_residual_norm": float(
                        candidate.nonlinear_result.diagnostics.initial_residual_norm
                    ),
                    "final_residual_norm": float(
                        candidate.nonlinear_result.diagnostics.final_residual_norm
                    ),
                    "final_step_norm": float(
                        candidate.nonlinear_result.diagnostics.final_step_norm
                    ),
                    "nonlinear_iterations": int(
                        candidate.nonlinear_result.diagnostics.iterations
                    ),
                    "linear_iterations": int(
                        candidate.nonlinear_result.diagnostics.linear_iterations
                    ),
                    "final_linear_status": int(
                        candidate.nonlinear_result.diagnostics.final_linear_status
                    ),
                    "reaction_pulling_N": diagnostics.reaction_pulling_N.tolist(),
                    "source_nearest_qp_force_N": diagnostics.source_boundary_force_N[
                        1
                    ].tolist(),
                    "interface_displacement_jump_l2_m": float(
                        diagnostics.interface_displacement_jump_l2_m
                    ),
                    "interface_traction_jump_l2_Pa": float(
                        diagnostics.interface_traction_jump_l2_Pa
                    ),
                    "interface_power_defect_W": float(
                        diagnostics.interface_power_defect_W
                    ),
                    "work_energy_residual_J": float(diagnostics.work_energy_residual_J),
                    "scope": "idealized-muscle-aponeurosis;numerical-only;no-tendon",
                },
                sort_keys=True,
            )
        )
        if not bool(candidate.successful):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
