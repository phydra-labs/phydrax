#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp

import phydrax as phx


def _mhd_case(dimension: int):
    count = 6
    names = tuple("xyz"[:dimension])
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True)
            for _ in range(dimension)
        ),
        axis_names=names,
    ).prepare(jnp.stack((jnp.zeros(dimension), jnp.ones(dimension))))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    system = phx.equations.IdealMHDSystem(dimension)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        f"advanced-mhd-{dimension}d",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(names),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLDFluxPlan(),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    primitive = jnp.zeros(grid.shape + (8,))
    primitive = primitive.at[..., 0].set(1.0)
    primitive = primitive.at[..., 4].set(1.0)
    primitive = primitive.at[..., 5].set(0.2)
    full = system.primitive_to_conserved(primitive)
    magnetic = bridge.pack_normal_flux(
        tuple(primitive[..., 5 + axis] for axis in range(dimension))
    )
    reconstruction = phx.solver.advanced.MHDPrimitiveReconstructionPlan("weno_z")
    spatial = phx.discretization.UpwindConstrainedTransportPlan(
        dynamics,
        bridge,
        reconstruction=reconstruction,
        electromotive_plan=phx.solver.advanced.HLLUCTElectromotivePlan(),
    )
    integrator = phx.solver.ConstrainedMHDSSPRK3Plan(spatial, cfl=0.2)
    state = integrator.initialize(full, magnetic, step_size=1e-4)
    started = time.perf_counter()
    result = integrator.advance(state, 0.0, 1e-4)
    jax.block_until_ready(result.state.cell_state)
    duration = time.perf_counter() - started
    return {
        "accepted": bool(result.accepted),
        "duration_seconds": duration,
        "cell_defect": float(
            jnp.max(jnp.abs(result.state.cell_state - state.cell_state))
        ),
        "magnetic_defect": float(
            jnp.max(jnp.abs(result.state.magnetic_flux - state.magnetic_flux))
        ),
        "constraint_change": float(result.diagnostics.magnetic_constraint_change),
        "accepted_face_families": len(result.accepted_integrals.face_flux_integrals),
        "accepted_edge_values": int(
            result.accepted_integrals.edge_electromotive_integrals.size
        ),
    }


def main() -> None:
    curve = phx.equations.TabulatedCoolingCurve(
        jnp.asarray([0.0, 1.0, 2.0]),
        jnp.asarray([-2.0, -1.0, 1.0]),
        bounds_policy="power_law_extrapolate",
    )
    temperature = jnp.asarray([1.5, 8.0, 40.0])
    recovered = curve.temperature_from_cooling_coordinate(
        curve.cooling_coordinate(temperature)
    )
    background = phx.applications.cosmology.FLRWBackground(1.0, 0.3)
    report = {
        "mhd": {str(dimension): _mhd_case(dimension) for dimension in (1, 2, 3)},
        "cooling_coordinate_round_trip": float(jnp.max(jnp.abs(recovered - temperature))),
        "cosmology": {
            "hubble_at_one": float(background.hubble(1.0)),
            "drift_half_to_one": float(background.drift_factor(0.5, 1.0)),
            "kick_half_to_one": float(background.kick_factor(0.5, 1.0)),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
