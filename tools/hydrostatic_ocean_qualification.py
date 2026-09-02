#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

import jax.numpy as jnp

import phydrax as phx


def _ocean(*, case, shape):
    nx, ny, nz = shape
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(nx, periodic=True),
            phx.discretization.UniformCellAxisSpec(ny, periodic=True),
            phx.discretization.UniformCellAxisSpec(nz, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -10.0), (float(nx), float(ny), 0.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("hydrostatic",)
    ).prepare()
    depth = (
        jnp.broadcast_to(jnp.linspace(0.02, 10.0, nx)[:, None], (nx, ny))
        if case == "wetdry"
        else jnp.full((nx, ny), 10.0)
    )
    geometry = phx.discretization.TensorZHydrostaticGridPlan(
        discretization,
        depth,
        vertical_coordinate="partial-z" if case == "partial" else "zstar",
    ).prepare()
    freshwater = (
        phx.applications.ocean.FreshwaterVolumeFluxPlan(1.0e-5)
        if case in ("freshwater", "wetdry")
        else None
    )
    mixing = phx.applications.ocean.HydrostaticMixingPlan(
        "kpp" if case == "closure" else "prescribed",
        maximum_coefficient=1.0e-3,
    )
    ocean = phx.applications.ocean.HydrostaticPrimitiveEquationPlan(
        geometry,
        eos=(
            phx.applications.ocean.NonlinearSeawaterPolynomialEOS()
            if case in ("thermodynamics", "closure")
            else None
        ),
        mixing=mixing,
        freshwater=freshwater,
        external_mode="split-explicit" if case == "wetdry" else "implicit",
        wetting_and_drying=case == "wetdry",
        subcycle_policy=phx.applications.ocean.ExternalModeSubcyclePolicy.fixed(10),
    ).prepare()
    eta = jnp.zeros((nx, ny))
    if case == "wave":
        eta = jnp.broadcast_to(
            1.0e-3 * jnp.sin(2.0 * jnp.pi * jnp.arange(nx)[:, None] / nx),
            (nx, ny),
        )
    state = ocean.initialize_state(eta)
    return ocean, state


def run_case(case, shape, dt):
    ocean, state = _ocean(case=case, shape=shape)
    continuation = phx.applications.ocean.HydrostaticContinuationState.initialize(
        ocean, state
    )
    result = phx.applications.ocean.HydrostaticIMEXMidpointMethod(ocean).step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(dt),
        None,
    )
    final = result.accepted_state.state
    epoch = ocean.geometry.metric_epoch(final.eta)
    view = phx.applications.ocean.hydrostatic_diagnostic_view(
        ocean, result.accepted_state
    )
    report = {
        "case": case,
        "shape": list(shape),
        "successful": bool(result.successful),
        "minimum_depth": float(jnp.min(epoch.total_depth)),
        "volume": float(view.volume),
        "volume_residual": float(result.accepted_state.ledger.residual),
        "free_surface_energy": float(view.free_surface_energy),
        "finite_density": bool(jnp.all(jnp.isfinite(view.density))),
    }
    report["passed"] = (
        report["successful"]
        and report["minimum_depth"] >= 0.0
        and report["finite_density"]
        and abs(report["volume_residual"]) <= 1.0e-6
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "rest",
            "wave",
            "freshwater",
            "partial",
            "wetdry",
            "thermodynamics",
            "closure",
        ),
        default="rest",
    )
    parser.add_argument("--shape", default="6,4,3")
    parser.add_argument("--dt", type=float, default=0.01)
    arguments = parser.parse_args()
    shape = tuple(int(value) for value in arguments.shape.split(","))
    if len(shape) != 3 or any(value < 3 for value in shape):
        raise ValueError("Hydrostatic qualification shape needs three counts >= 3.")
    if arguments.dt <= 0.0:
        raise ValueError("Hydrostatic qualification dt must be positive.")
    print(
        json.dumps(
            run_case(arguments.case, shape, arguments.dt),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
