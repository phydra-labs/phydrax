#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _graph(case: Any) -> Any:
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


_DROP_CELLS = 24
_DROP_RADIUS = 0.3
_DROP_SURFACE_TENSION = 0.072


def _drop_fraction(cells: int, radius: float, samples: int = 16) -> np.ndarray:
    """Liquid fraction of the centered drop from ``samples²`` points per cell."""

    offsets = (np.arange(samples) + 0.5) / (samples * cells)
    x = (np.arange(cells)[:, None] / cells + offsets).reshape(-1)
    inside = (x[:, None] - 0.5) ** 2 + (x[None, :] - 0.5) ** 2 < radius**2
    return inside.reshape(cells, samples, cells, samples).mean(axis=(1, 3))


def _two_phase() -> Any:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(_DROP_CELLS, periodic=True),
            phx.discretization.UniformCellAxisSpec(_DROP_CELLS, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    material = phx.applications.two_phase_flow.TwoPhaseMaterialPlan(
        liquid_density=1000.0,
        gas_density=10.0,
        surface_tension=_DROP_SURFACE_TENSION,
    )
    two_phase = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan(
        discretization, material, maximum_iterations=4000
    ).prepare()
    alpha = jnp.asarray(_drop_fraction(_DROP_CELLS, _DROP_RADIUS))
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(two_phase)
    return two_phase, method, method.initial_continuation(two_phase.initial_state(alpha))


def _passive_tracer() -> Any:
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


def run_case(case: Any, dt: Any) -> Any:
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
            "passed": bool(
                result.success
                and result.donor_bounded
                and jnp.isfinite(error)
                and jnp.isfinite(result.integral_defect)
                and jnp.isfinite(result.maximum_displacement_cell_widths)
                and error <= 5.0e-2
                and jnp.abs(result.integral_defect) <= 1.0e-8
                and result.limiter_active_count >= 0
            ),
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
        candidate = result.candidate_state
        final_volume = jnp.sum(candidate.state.liquid_content)
        evidence = candidate.evidence
        # Laplace jump of the static drop: mean pressure two cells inside the
        # interface minus two cells outside, against sigma / R.
        radius = np.linalg.norm(
            np.asarray(two_phase.plan.discretization.cell_centers) - 0.5, axis=-1
        )
        pressure = np.asarray(candidate.pressure)
        band = 2.0 / _DROP_CELLS
        measured_jump = float(
            pressure[radius < _DROP_RADIUS - band].mean()
            - pressure[radius > _DROP_RADIUS + band].mean()
        )
        laplace_jump = _DROP_SURFACE_TENSION / _DROP_RADIUS
        return {
            "case": case,
            "successful": bool(result.successful),
            "liquid_volume_defect": float(final_volume - initial_volume),
            "alpha_minimum": float(evidence.alpha_minimum),
            "alpha_maximum": float(evidence.alpha_maximum),
            "divergence_residual": float(evidence.divergence_residual),
            "topology_events": int(evidence.topology_event_count),
            "laplace_jump": laplace_jump,
            "measured_pressure_jump": measured_jump,
            "curvature_pressure_jump": float(evidence.capillary_pressure_jump),
            "parasitic_velocity": float(evidence.parasitic_velocity),
            "curvature_valid_cells": int(evidence.curvature_valid_count),
            "curvature_fallback_cells": int(evidence.curvature_fallback_count),
            "curvature_underresolved_cells": int(evidence.curvature_underresolved_count),
            "passed": bool(
                result.successful
                and jnp.isfinite(final_volume)
                and jnp.isfinite(evidence.alpha_minimum)
                and jnp.isfinite(evidence.alpha_maximum)
                and jnp.isfinite(evidence.divergence_residual)
                and abs(float(final_volume - initial_volume)) <= 1.0e-8
                and evidence.alpha_minimum >= 0.0
                and evidence.alpha_maximum <= 1.0
                and jnp.abs(evidence.divergence_residual) <= 1.0e-8
                and evidence.topology_event_count >= 0
                and evidence.curvature_underresolved_count == 0
                and abs(measured_jump - laplace_jump) <= 5.0e-2 * laplace_jump
                and abs(float(evidence.capillary_pressure_jump) - laplace_jump)
                <= 5.0e-2 * laplace_jump
            ),
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
            "passed": bool(
                result.evidence.successful
                and result.evidence.conservative
                and jnp.isfinite(result.evidence.momentum_defect)
                and abs(float(result.evidence.momentum_defect)) <= 1.0e-8
                and all(
                    jnp.isfinite(value) and jnp.abs(value) <= 1.0e-8
                    for value in result.evidence.scalar_content_defect.values()
                )
            ),
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
        "passed": bool(
            result.successful
            and jnp.all(
                jnp.isfinite(
                    jnp.asarray(
                        (
                            ledger.volume_change,
                            ledger.total_energy_residual,
                            ledger.capillary_dual_residual,
                            ledger.wave_work,
                            ledger.sponge_dissipation,
                        )
                    )
                )
            )
            and abs(float(ledger.volume_change)) <= 1.0e-7
            and abs(float(ledger.total_energy_residual)) <= 1.0e-7
            and abs(float(ledger.capillary_dual_residual)) <= 1.0e-7
            and float(ledger.sponge_dissipation) >= 0.0
        ),
    }


def main() -> Any:
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
    report = run_case(arguments.case, arguments.dt)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
