#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

import jax.numpy as jnp

import phydrax as phx


def _graph(case):
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
    wave = None
    if case == "wave":
        provider = phx.equations.IncidentWavePlan(
            (phx.equations.WaveComponent(1.0e-4, 1.0),), 1.0
        )
        wave = phx.applications.hydrodynamics.WaveForcingPlan(
            provider,
            jnp.zeros(shape).at[:2].set(0.5),
            jnp.zeros(shape).at[-2:].set(0.25),
            active_gain=0.1,
        )
    surface = phx.applications.hydrodynamics.GraphSurfaceALEPlan(
        reference, jnp.full(shape[:2], -1.0), maximum_slope=0.8
    )
    hydro = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEPlan(
        surface,
        surface_tension=0.072 if case == "capillary" else 0.0,
        wave=wave,
        coupling_iterations=5,
        coupling_tolerance=1.0e-7,
    ).prepare()
    state = hydro.initial_state(jnp.zeros(shape[:2]))
    continuation = (
        phx.applications.hydrodynamics.FreeSurfaceALEContinuationState.initialize(state)
    )
    return hydro, continuation


def _two_phase():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    material = phx.applications.two_phase_flow.TwoPhaseMaterialPlan(
        liquid_density=1000.0,
        gas_density=10.0,
        surface_tension=0.072,
    )
    two_phase = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan(
        discretization, material
    ).prepare()
    x = (jnp.arange(8) + 0.5) / 8
    y = (jnp.arange(8) + 0.5) / 8
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    alpha = jnp.where((xx - 0.5) ** 2 + (yy - 0.5) ** 2 < 0.2**2, 1.0, 0.0)
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(two_phase)
    return two_phase, method, method.initial_continuation(two_phase.initial_state(alpha))


def _passive_tracer():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(32, periodic=True),
            phx.discretization.UniformCellAxisSpec(32, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    mac = phx.discretization.MACOperatorPlan(discretization).prepare()
    space = grid.field_space(
        "passive-tracer",
        entity_layout=discretization.cell_layout,
        dtype=mac.pressure_space.dtype,
        representation="point_value",
    )
    transport = phx.discretization.MACPassiveTracerMacCormackPlan(
        mac,
        space,
    ).prepare()
    center = jnp.asarray((0.35, 0.4))
    values = jnp.exp(
        -120.0 * jnp.sum((discretization.cell_centers - center) ** 2, axis=-1)
    )
    velocity = (
        jnp.full(discretization.face_layouts[0].shape, 0.25),
        jnp.full(discretization.face_layouts[1].shape, -0.1),
    )
    return discretization, transport, values, velocity, center


def run_case(case, dt):
    if case == "passive-tracer":
        discretization, transport, values, velocity, center = _passive_tracer()
        result = transport.advance(values, velocity, jnp.asarray(dt))
        translated = center + jnp.asarray((0.25, -0.1)) * dt
        error = jnp.sqrt(
            jnp.mean(
                (
                    result.values
                    - jnp.exp(
                        -120.0
                        * jnp.sum(
                            (discretization.cell_centers - translated) ** 2,
                            axis=-1,
                        )
                    )
                )
                ** 2
            )
        )
        return {
            "case": case,
            "successful": bool(result.success),
            "donor_bounded": bool(result.donor_bounded),
            "l2_error": float(error),
            "integral_defect": float(result.integral_defect),
            "limiter_cells": int(result.limiter_active_count),
            "maximum_displacement_cells": float(result.maximum_displacement_cell_widths),
            "passed": bool(result.success) and float(error) <= 5.0e-2,
        }
    if case == "two-phase":
        two_phase, method, continuation = _two_phase()
        initial_volume = jnp.sum(continuation.state.liquid_content)
        result = method.step(
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0),
            continuation,
            jnp.asarray(dt),
            None,
        )
        final_volume = jnp.sum(result.accepted_state.state.liquid_content)
        evidence = result.accepted_state.evidence
        return {
            "case": case,
            "successful": bool(result.successful),
            "liquid_volume_defect": float(final_volume - initial_volume),
            "alpha_minimum": float(evidence.alpha_minimum),
            "alpha_maximum": float(evidence.alpha_maximum),
            "divergence_residual": float(evidence.divergence_residual),
            "topology_events": int(evidence.topology_event_count),
            "passed": bool(result.successful)
            and abs(float(final_volume - initial_volume)) <= 1.0e-8,
        }
    hydro, continuation = _graph(case)
    if case == "rezone":
        rezone = phx.applications.hydrodynamics.FreeSurfaceRezonePlan(1.4)
        result = rezone.rezone(hydro, continuation)
        return {
            "case": case,
            "successful": bool(result.evidence.successful),
            "scalar_defect": float(
                max(
                    tuple(
                        jnp.abs(value)
                        for value in result.evidence.scalar_content_defect.values()
                    )
                )
            ),
            "momentum_defect": float(result.evidence.momentum_defect),
            "mesh_epoch": int(result.state.mesh_epoch),
            "passed": bool(result.evidence.conservative),
        }
    method = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEMethod(hydro)
    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(dt),
        None,
    )
    ledger = result.accepted_state.ledger
    return {
        "case": case,
        "successful": bool(result.successful),
        "volume_change": float(ledger.volume_change),
        "energy_residual": float(ledger.total_energy_residual),
        "capillary_dual_residual": float(ledger.capillary_dual_residual),
        "wave_work": float(ledger.wave_work),
        "sponge_dissipation": float(ledger.sponge_dissipation),
        "passed": bool(result.successful) and abs(float(ledger.volume_change)) <= 1.0e-7,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "baseline",
            "capillary",
            "wave",
            "rezone",
            "two-phase",
            "passive-tracer",
        ),
        default="baseline",
    )
    parser.add_argument("--dt", type=float, default=0.001)
    arguments = parser.parse_args()
    if arguments.dt <= 0.0:
        raise ValueError("Advanced hydrodynamic qualification dt must be positive.")
    print(
        json.dumps(
            run_case(arguments.case, arguments.dt),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
