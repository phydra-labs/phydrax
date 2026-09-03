#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from typing import Any

import jax.numpy as jnp

from phydrax.applications.cardiovascular.circulation._closed_loop import (
    biventricular_closed_loop,
    pulmonary_closed_loop,
    systemic_closed_loop,
)
from phydrax.applications.cardiovascular.circulation._components import (
    rc_pressure_transient,
    Resistance,
    WindkesselRCR,
)
from phydrax.applications.cardiovascular.circulation._coronary import (
    coronary_closed_loop,
    CoronaryCirculation,
)
from phydrax.applications.cardiovascular.circulation._network import (
    initialize_consistent_state,
    prepare_consistent_initialization,
)
from phydrax.applications.cardiovascular.circulation._periodic import (
    pressure_volume_work,
)
from phydrax.applications.cardiovascular.circulation._valves import (
    ComplementarityValve,
    EventValve,
    SmoothValve,
)
from phydrax.dynamics import analyze_dae_structure, DAEStructuralPolicy, TimeGrid
from phydrax.nonlinear import NonlinearTermination
from phydrax.solver import (
    BDFMethod,
    DAESolvePolicy,
    DifferentialAlgebraicProblem,
    solve_dae,
)


def _trajectory(compilation, states, variable_name: str):
    index = compilation.analysis.variable_names.index(variable_name)
    return states[:, index]


def _passive_dissipation(model: Any, compilation, states):
    total = jnp.zeros((states.shape[0],), dtype=states.dtype)
    for component in model.network.components:
        if isinstance(component, Resistance):
            flow = _trajectory(compilation, states, f"{component.name}.flow_out")
            total = total + component.resistance * flow * flow
        elif isinstance(component, WindkesselRCR):
            flow_in = _trajectory(compilation, states, f"{component.name}.flow_in")
            flow_out = _trajectory(compilation, states, f"{component.name}.flow_out")
            total = (
                total
                + component.proximal_resistance * flow_in * flow_in
                + component.distal_resistance * flow_out * flow_out
            )
        elif isinstance(
            component,
            (SmoothValve, ComplementarityValve, EventValve),
        ):
            pressure_in = _trajectory(
                compilation, states, f"{component.name}.pressure_in"
            )
            pressure_out = _trajectory(
                compilation, states, f"{component.name}.pressure_out"
            )
            flow = _trajectory(compilation, states, f"{component.name}.flow_out")
            total = total + (pressure_in - pressure_out) * flow
    return total


def _production_loop_metrics(name: str, model: Any) -> dict[str, object]:
    prepared = prepare_consistent_initialization(model.network)
    initialized = initialize_consistent_state(prepared)
    termination = NonlinearTermination(
        absolute_residual=2.0e-2,
        relative_residual=0.0,
        maximum_steps=64,
    )
    policy = DAESolvePolicy(
        method=BDFMethod(1),
        nonlinear_termination=termination,
        initialization_termination=termination,
        failure="status",
    )
    times = jnp.linspace(0.0, 0.01, 3)
    problem = DifferentialAlgebraicProblem(
        prepared.compilation.system,
        initialized.state,
        initial_state_rate=initialized.state_rate,
        problem_id=f"cardiovascular-circulation-qualification:{name}",
    )
    solution = solve_dae(
        problem,
        TimeGrid(times, time_id=f"cardiovascular-circulation:{name}:short-runtime"),
        policy=policy,
    )
    assembly_residual = prepared.compilation.residual_audit(
        jnp.asarray(0.0), initialized.state, initialized.state_rate, None
    )
    structure = analyze_dae_structure(
        model.network.source,
        DAEStructuralPolicy(2, 128, tearing="automatic"),
    )
    runtime_finite = bool(
        jnp.all(jnp.isfinite(solution.states))
        & jnp.all(jnp.isfinite(solution.state_rates))
        & jnp.all(jnp.isfinite(solution.residual_norm))
    )
    if bool(solution.successful) and runtime_finite:
        volumes = jnp.stack(
            tuple(
                _trajectory(prepared.compilation, solution.states, storage_id)
                for storage_id in model.network.storage_ids
            ),
            axis=-1,
        )
        total_volume = jnp.sum(volumes, axis=-1)
        volume_scale = jnp.maximum(jnp.abs(total_volume[0]), 1.0)
        volume_residual = float(
            jnp.max(jnp.abs(total_volume - total_volume[0])) / volume_scale
        )
        dissipation = _passive_dissipation(model, prepared.compilation, solution.states)
        chamber_name = (
            model.bed_name
            if isinstance(model, CoronaryCirculation)
            else model.chamber_names[0]
        )
        pressure = _trajectory(
            prepared.compilation,
            solution.states,
            f"{chamber_name}.pressure_in",
        )
        chamber_volume = _trajectory(
            prepared.compilation,
            solution.states,
            f"{chamber_name}.volume",
        )
        segment_work = float(pressure_volume_work(pressure, chamber_volume))
        minimum_dissipation = float(jnp.min(dissipation))
        runtime_residual = float(jnp.max(solution.residual_norm))
    else:
        volume_residual = None
        minimum_dissipation = None
        segment_work = None
        runtime_residual = None
    return {
        "component_count": len(model.network.components),
        "variable_count": len(structure.variable_names),
        "equation_count": len(structure.equation_names),
        "structural_success": bool(structure.successful),
        "initialization_success": bool(initialized.evidence.successful),
        "assembled_residual_norm": float(jnp.max(jnp.abs(assembly_residual))),
        "runtime_success": bool(solution.successful),
        "runtime_max_residual_norm": runtime_residual,
        "runtime_finite": runtime_finite,
        "total_volume_relative_residual": volume_residual,
        "minimum_passive_dissipation_kPa_mm3_per_ms": minimum_dissipation,
        "chamber_segment_work_kPa_mm3": segment_work,
    }


def main() -> None:
    time = jnp.linspace(0.0, 10.0, 101)
    analytic = rc_pressure_transient(time, 2.0, 12.0, 2.0, 1.5)
    analytic_reference = 12.0 - 10.0 * jnp.exp(-time / 3.0)
    transient_error = float(jnp.max(jnp.abs(analytic - analytic_reference)))

    reference_loops = {
        "systemic": systemic_closed_loop(),
        "pulmonary": pulmonary_closed_loop(),
        "biventricular": biventricular_closed_loop(),
        "coronary": coronary_closed_loop(),
    }
    loop_metrics = {
        name: _production_loop_metrics(name, model)
        for name, model in reference_loops.items()
    }

    valve = EventValve(
        "qualification_valve",
        0.01,
        100.0,
        opening_pressure=0.1,
        closing_pressure=-0.1,
        minimum_dwell_time=2.0,
    )
    opening = valve.propose_event(0.0, 1.0)
    rejected = valve.commit_event(opening, accept=False)
    opened = valve.commit_event(opening)
    closing = opened.propose_event(2.0, -1.0)
    event_identity_valid = bool(
        opening.event_required
        and closing.event_required
        and opening.source_state_id == valve.state.state_id
        and rejected.state.state_id == valve.state.state_id
    )

    loop_passed = all(
        metrics["structural_success"]
        and metrics["initialization_success"]
        and metrics["runtime_success"]
        and metrics["runtime_finite"]
        and metrics["assembled_residual_norm"] < 1.0e-8
        and metrics["runtime_max_residual_norm"] <= 2.0e-2
        and metrics["total_volume_relative_residual"] < 1.0e-8
        and metrics["minimum_passive_dissipation_kPa_mm3_per_ms"] >= -1.0e-9
        for metrics in loop_metrics.values()
    )
    passed = bool(transient_error < 1.0e-12 and event_identity_valid and loop_passed)
    print(
        json.dumps(
            {
                "campaign": "cardiovascular-circulation",
                "passed": passed,
                "analytic_transient_max_error": transient_error,
                "event_source_identity_valid": event_identity_valid,
                "reference_loops": loop_metrics,
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
