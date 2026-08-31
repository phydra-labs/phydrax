#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

import phydrax as phx


def _gravity_mode():
    count = 32
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "gravity-mode",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.HLLCFluxPlan(),
        ),
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics, phx.discretization.FluxPositivityPlan()
    )
    gravity = phx.solver.NewtonianSelfGravityPlan(0.1).prepare(
        phx.solver.prepare_balance_law_transport(runtime)
    )
    x = grid.structured_axes[0].interval_centers
    density = 1.0 + 0.01 * jnp.sin(2.0 * jnp.pi * x)
    potential, _, acceleration, solved = gravity.solve_density(density)
    return {
        "residual": float(solved.residual_norm),
        "compatibility": float(solved.compatibility_residual),
        "gauge": float(jnp.abs(jnp.mean(potential))),
        "force_mean": float(jnp.abs(jnp.mean(density * acceleration[..., 0]))),
    }


def _cochain_and_hlld():
    count = 3
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(3)
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    face_shape = (count, count, count)
    magnetic_flux = bridge.pack_face_flux(
        (jnp.full(face_shape, 0.2), jnp.zeros(face_shape), jnp.zeros(face_shape))
    )
    system = phx.equations.IdealMHDSystem(3)
    primitive = jnp.zeros(face_shape + (8,))
    primitive = primitive.at[..., 0].set(1.0)
    primitive = primitive.at[..., 4].set(1.0)
    primitive = primitive.at[..., 5].set(0.2)
    conservative = system.primitive_to_conserved(primitive)
    result = phx.discretization.HLLDFluxPlan().face_flux(
        system, conservative, conservative, 0
    )
    return {
        "magnetic_constraint": float(
            jnp.max(jnp.abs(bridge.exterior_derivative(2, magnetic_flux)), initial=0.0)
        ),
        "hlld_flux_defect": float(
            jnp.max(jnp.abs(result.normal_flux - system.physical_flux(conservative, 0)))
        ),
        "hlld_fallback_count": int(jnp.sum(result.fallback_activated)),
    }


def _cooling_curve():
    curve = phx.equations.TabulatedCoolingCurve(
        jnp.asarray([1.0, 2.0, 3.0]),
        jnp.asarray([-2.0, -1.0, 0.0]),
        bounds_policy="power_law_extrapolate",
    )
    evaluated = curve.evaluate(jnp.asarray([10.0, 100.0, 1000.0]))
    return {
        "supported": bool(jnp.all(evaluated.supported)),
        "minimum_rate": float(jnp.min(evaluated.rate)),
        "maximum_rate": float(jnp.max(evaluated.rate)),
    }


def main() -> None:
    report = {
        "backend": jax.default_backend(),
        "gravity": _gravity_mode(),
        "mhd": _cochain_and_hlld(),
        "cooling": _cooling_curve(),
    }
    report["passed"] = bool(
        report["gravity"]["residual"] < 1e-8
        and report["gravity"]["gauge"] < 1e-10
        and report["mhd"]["magnetic_constraint"] < 1e-12
        and report["mhd"]["hlld_flux_defect"] < 1e-9
        and report["mhd"]["hlld_fallback_count"] == 0
        and report["cooling"]["supported"]
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
