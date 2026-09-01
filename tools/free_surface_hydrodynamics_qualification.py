#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

import jax.numpy as jnp

import phydrax as phx


def _model(case):
    shape = (4, 4, 3)
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(3, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (4.0, 4.0, 0.0))))
    reference = phx.discretization.FiniteVolumePlan(
        grid, component_names=("hydrodynamics",)
    ).prepare()
    surface = phx.applications.hydrodynamics.GraphSurfaceALEPlan(
        reference,
        jnp.full(shape[:2], -1.0),
        maximum_slope=0.5,
        maximum_iterations=150,
    )
    hydro = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEPlan(
        surface,
        coupling_iterations=5,
        coupling_tolerance=1.0e-7,
    ).prepare()
    eta = jnp.zeros(shape[:2])
    if case == "wave":
        eta = jnp.broadcast_to(
            1.0e-4 * jnp.sin(2.0 * jnp.pi * jnp.arange(4)[:, None] / 4),
            shape[:2],
        )
    state = hydro.initial_state(eta)
    if case == "scalar":
        geometry = hydro.surface.geometry(0.0, eta, jnp.zeros_like(eta))
        state = phx.applications.hydrodynamics.FreeSurfaceALEState(
            state.eta,
            state.momentum,
            {"uniform": 2.0 * geometry.cell_volumes},
        )
    return hydro, state


def run_case(case, dt):
    hydro, state = _model(case)
    if case == "gcl":
        rate = jnp.full_like(state.eta, 1.0e-4)
        evidence = hydro.surface.geometry_evidence(state.eta, rate)
        target = rate * hydro.surface.horizontal_area
        kinematic = hydro.surface.solve_eta_rate(state.eta, target)
        return {
            "case": case,
            "geometry_valid": bool(evidence.valid),
            "gcl_residual": float(evidence.volume_gcl_residual),
            "kinematic_residual": float(kinematic.residual_norm),
            "passed": bool(evidence.valid) and bool(kinematic.converged),
        }
    continuation = (
        phx.applications.hydrodynamics.FreeSurfaceALEContinuationState.initialize(state)
    )
    result = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEMethod(hydro).step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(dt),
        None,
    )
    view = hydro.view(
        result.accepted_state.state,
        result.accepted_state.eta_rate,
    )
    scalar_error = (
        0.0
        if case != "scalar"
        else float(jnp.max(jnp.abs(view.scalars["uniform"] - 2.0)))
    )
    report = {
        "case": case,
        "successful": bool(result.successful),
        "volume_change": float(result.accepted_state.ledger.volume_change),
        "divergence_residual": float(result.accepted_state.ledger.divergence_residual),
        "kinematic_residual": float(result.accepted_state.ledger.kinematic_residual),
        "nonlinear_residual": float(
            result.accepted_state.ledger.nonlinear_stage_residual
        ),
        "scalar_error": scalar_error,
    }
    report["passed"] = (
        report["successful"]
        and abs(report["volume_change"]) <= 1.0e-7
        and report["divergence_residual"] <= 1.0e-7
        and report["kinematic_residual"] <= 1.0e-7
        and scalar_error <= 1.0e-7
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=("rest", "wave", "gcl", "scalar"), default="rest"
    )
    parser.add_argument("--dt", type=float, default=0.002)
    arguments = parser.parse_args()
    if arguments.dt <= 0.0:
        raise ValueError("Hydrodynamic qualification dt must be positive.")
    print(
        json.dumps(
            run_case(arguments.case, arguments.dt),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
