#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualification campaigns for resolved bubbly flow.

Each campaign runs the full two-phase VOF step with its bubbly-flow coupling.
It compares against an analytic reference and records the metrics, the gate
decision and the runtime identity. Every campaign is bounded (one short run
per resolution) and writes plain JSON.

- ``breathing-closed``: a liquid slab between two closed gas slabs. The
  linear frequency is
  ``omega^2 = kappa p0 / (rho_l L_l) (1/L_1 + 1/L_2)`` (isothermal and
  adiabatic).
- ``breathing-vented``: a closed gas slab under a liquid slab, vented above,
  with ``omega^2 = kappa p0 / (rho_l L_l L_g)``.
- ``boyle-rise``: an isothermal bubble rising in a vented column. The gas
  volume follows ``V p = const`` with the hydrostatic pressure at the bubble
  centroid plus the Laplace jump.
- ``thermocapillary``: two bounded refinements of a drop in an imposed linear
  temperature field, compared with the 2D Young–Goldstein–Block Stokes-limit
  velocity ``U = -sigma_T G a / (4 (mu + mu'))`` (``k' = k``).

Usage, from the repository root:
`PYTHONPATH=. python tools/bubbly_flow_qualification.py [--campaign NAME] [--output PATH]`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
import phydrax.bubble_dynamics as bubble_dynamics
from phydrax.qualification import QualificationRuntimeIdentity


two_phase_api = phx.applications.two_phase_flow
Record = dict[str, Any]


def _digest(payload: Record, /) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _runtime_identity() -> Record:
    identity = QualificationRuntimeIdentity(
        _digest(
            {
                "kind": "bubbly-flow-qualification-build",
                "phydrax": importlib.metadata.version("phydrax"),
                "path": str(Path(phx.__file__).resolve().parent),
            }
        ),
        _digest(
            {
                "kind": "bubbly-flow-qualification-environment",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "jax": jax.__version__,
                "numpy": np.__version__,
            }
        ),
        jax.default_backend(),
        f"processes-{jax.process_count()}-devices-{jax.device_count()}",
        str(jnp.asarray(0.0).dtype),
    )
    return dict(identity.to_record())


def _discretization(
    shape: tuple[int, int], upper: tuple[float, float], periodic: tuple[bool, bool]
) -> phx.discretization.FiniteVolumeDiscretization:
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=flag)
            for count, flag in zip(shape, periodic, strict=True)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), upper)))
    return phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()


def _disk_fraction(
    shape: tuple[int, int],
    upper: tuple[float, float],
    center: tuple[float, float],
    radius: float,
    /,
) -> np.ndarray:
    samples = 4
    content = np.zeros(shape, dtype=np.float64)
    lower_x = np.arange(shape[0]) * upper[0] / shape[0]
    lower_y = np.arange(shape[1]) * upper[1] / shape[1]
    offsets_x = (np.arange(samples) + 0.5) * upper[0] / (samples * shape[0])
    offsets_y = (np.arange(samples) + 0.5) * upper[1] / (samples * shape[1])
    for offset_x in offsets_x:
        for offset_y in offsets_y:
            x, y = np.meshgrid(
                lower_x + offset_x, lower_y + offset_y, indexing="ij"
            )
            content += (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius**2
    return content / samples**2


def _law(kind: str, /) -> bubble_dynamics.AbstractBubbleCompartmentGasLaw:
    if kind == "isothermal":
        return bubble_dynamics.IsothermalIdealBubbleGasLaw(1.4)
    return bubble_dynamics.CaloricIdealBubbleGasLaw(1.4)


def _period(times: np.ndarray, signal: np.ndarray, /) -> float:
    """Mean period from upward mean crossings (linear interpolation)."""

    centred = signal - np.mean(signal)
    crossings = []
    for index in range(1, centred.size):
        if centred[index - 1] < 0.0 <= centred[index]:
            fraction = -centred[index - 1] / (centred[index] - centred[index - 1])
            crossings.append(
                times[index - 1] + fraction * (times[index] - times[index - 1])
            )
    if len(crossings) < 2:
        return float("nan")
    return float(np.mean(np.diff(crossings)))


def _breathing(kind: str, cells: int, vented: bool, /) -> Record:
    """Slab breathing run: returns measured and analytic periods."""

    liquid_density = 1000.0
    pressure = 1000.0
    discretization = _discretization((4, cells), (0.25, 1.0), (True, False))
    material = two_phase_api.TwoPhaseMaterialPlan(
        liquid_density=liquid_density,
        gas_density=1.0,
        liquid_viscosity=0.0,
        gas_viscosity=0.0,
    )
    solve_tolerance = 1.0e-11
    maximum_iterations = 4000
    two_phase = two_phase_api.IncompressibleTwoPhaseVOFPlan(
        discretization,
        material,
        tolerance=solve_tolerance,
        maximum_iterations=maximum_iterations,
    ).prepare()
    y = np.asarray(discretization.cell_centers)[..., 1]
    lower, upper = 0.25, 0.75
    alpha = np.where((y > lower) & (y < upper), 1.0, 0.0)
    identity = two_phase_api.BubbleComponentPlan(
        two_phase,
        component_capacity=4,
        maximum_rounds=64,
        pair_capacity=8,
        vent_sides=((1, "upper"),) if vented else (),
    )
    compartments = two_phase_api.BubbleCompartmentPlan(
        _law(kind),
        bubble_dynamics.BubbleEnvironment(pressure, 300.0),
        capacity=2,
        dimension=2,
    )
    plan = two_phase_api.BubblyFlowPlan(
        two_phase, identity, compartments=compartments, atmosphere_pressure=pressure
    )
    epsilon = 0.02
    start = (
        np.asarray([pressure * (1.0 + epsilon), pressure, 0.0, 0.0])
        if vented
        else np.asarray(
            [pressure * (1.0 + epsilon), pressure * (1.0 - epsilon), 0.0, 0.0]
        )
    )
    bubbles = plan.initial_state(jnp.asarray(alpha), compartment_pressure=start)
    initial_compartments = bubbles.compartments
    if initial_compartments is None:
        raise RuntimeError("Breathing run did not initialize compartments.")
    state = two_phase.initial_state(jnp.asarray(alpha))
    method = two_phase_api.IncompressibleTwoPhaseVOFMethod(two_phase, bubbles=plan)
    continuation = method.initial_continuation(state, bubbles=bubbles)
    exponent = 1.0 if kind == "isothermal" else 1.4
    gas, liquid = lower, upper - lower
    stiffness = (
        exponent * pressure / (liquid_density * liquid * gas)
        if vented
        else exponent * pressure / (liquid_density * liquid) * (2.0 / gas)
    )
    omega = float(np.sqrt(stiffness))
    period = 2.0 * np.pi / omega
    step = period / 200.0
    steps = 600
    times = [0.0]
    volumes = [float(initial_compartments.volume[0])]

    def observe(time: float, current: two_phase_api.TwoPhaseContinuationState) -> None:
        current_bubbles = current.bubbles
        if current_bubbles is None or current_bubbles.compartments is None:
            raise RuntimeError("Breathing run lost its compartment registry.")
        times.append(time)
        volumes.append(float(current_bubbles.compartments.volume[0]))

    run = two_phase_api.run_bubbly_flow(
        method, continuation, steps=steps, step_size=step, observer=observe
    )
    measured = _period(np.asarray(times), np.asarray(volumes))
    evidence = run.continuation.bubble_evidence
    if evidence is None:
        raise RuntimeError("Breathing run did not produce bubble evidence.")
    return {
        "law": kind,
        "cells": cells,
        "vented": vented,
        "status": run.status.name,
        "projection_status": phx.solver.MACCompartmentProjectionStatus(
            int(evidence.projection_status)
        ).name,
        "solve_tolerance": solve_tolerance,
        "maximum_iterations": maximum_iterations,
        "analytic_period": period,
        "measured_period": measured,
        "relative_error": abs(measured - period) / period,
        "steps": len(times) - 1,
        "work_identity_residual": float(evidence.work_identity_residual),
        "eos_residual": float(evidence.eos_residual),
        "compartment_pressure_residual": float(evidence.compartment_pressure_residual),
    }


def breathing_closed() -> Record:
    rows = [
        _breathing(kind, cells, False)
        for kind in ("isothermal", "caloric")
        for cells in (16, 32)
    ]
    return {
        "reference": "omega^2 = kappa p0 / (rho_l L_l) (1/L_1 + 1/L_2)",
        "rows": rows,
        "passed": all(
            row["status"] == "COMPLETED" and row["relative_error"] < 0.03 for row in rows
        ),
    }


def breathing_vented() -> Record:
    rows = [
        _breathing(kind, cells, True)
        for kind in ("isothermal", "caloric")
        for cells in (16, 32)
    ]
    return {
        "reference": "omega^2 = kappa p0 / (rho_l L_l L_g)",
        "rows": rows,
        "passed": all(
            row["status"] == "COMPLETED" and row["relative_error"] < 0.03 for row in rows
        ),
    }


def boyle_rise() -> Record:
    """Bounded isothermal rise probe with a vented atmosphere and Boyle ledger."""

    cells = 16
    pressure = 100.0
    discretization = _discretization((cells, 2 * cells), (1.0, 2.0), (False, False))
    material = two_phase_api.TwoPhaseMaterialPlan(
        liquid_density=1.0,
        gas_density=0.1,
        liquid_viscosity=0.1,
        gas_viscosity=0.01,
    )
    two_phase = two_phase_api.IncompressibleTwoPhaseVOFPlan(
        discretization,
        material,
        gravity=(0.0, -1.0),
        hydrostatic_reference=(0.5, 2.0),
        reference_pressure=pressure,
        maximum_iterations=2000,
    ).prepare()
    centers = np.asarray(discretization.cell_centers)
    bubble = _disk_fraction(
        (cells, 2 * cells), (1.0, 2.0), (0.5, 0.45), 0.16
    )
    atmosphere = centers[..., 1] > 1.75
    alpha = 1.0 - np.maximum(bubble, atmosphere.astype(np.float64))
    identity = two_phase_api.BubbleComponentPlan(
        two_phase,
        component_capacity=6,
        maximum_rounds=128,
        pair_capacity=16,
        vent_sides=((1, "upper"),),
    )
    compartments = two_phase_api.BubbleCompartmentPlan(
        _law("isothermal"),
        bubble_dynamics.BubbleEnvironment(pressure, 300.0),
        capacity=4,
        dimension=2,
    )
    plan = two_phase_api.BubblyFlowPlan(
        two_phase,
        identity,
        compartments=compartments,
        atmosphere_pressure=pressure,
    )
    bubbles = plan.initial_state(
        jnp.asarray(alpha), compartment_pressure=pressure + 1.55
    )
    initial_compartments = bubbles.compartments
    if initial_compartments is None:
        raise RuntimeError("Boyle-rise run did not initialize compartments.")
    method = two_phase_api.IncompressibleTwoPhaseVOFMethod(two_phase, bubbles=plan)
    continuation = method.initial_continuation(
        two_phase.initial_state(jnp.asarray(alpha)), bubbles=bubbles
    )
    active = np.asarray(initial_compartments.active)
    initial_boyle = np.asarray(
        initial_compartments.pressure * initial_compartments.volume
    )[active]
    run = two_phase_api.run_bubbly_flow(
        method, continuation, steps=4, step_size=2.0e-4
    )
    final_bubbles = run.continuation.bubbles
    if final_bubbles is None or final_bubbles.compartments is None:
        raise RuntimeError("Boyle-rise run lost its compartment registry.")
    final_active = np.asarray(final_bubbles.compartments.active)
    final_boyle = np.asarray(
        final_bubbles.compartments.pressure * final_bubbles.compartments.volume
    )[final_active]
    count = min(initial_boyle.size, final_boyle.size)
    residual = (
        float("inf")
        if count == 0
        else float(
            np.max(
                np.abs(final_boyle[:count] - initial_boyle[:count])
                / np.maximum(np.abs(initial_boyle[:count]), 1.0)
            )
        )
    )
    return {
        "reference": "isothermal Boyle invariant p V = constant",
        "status": run.status.name,
        "steps": run.steps,
        "relative_boyle_residual": residual,
        "passed": run.completed and residual < 1.0e-8,
    }


def _thermocapillary_resolution(cells: int, /) -> Record:
    radius = 0.2
    viscosity = 1.0
    slope = -1.0e-2
    gradient = 1.0
    discretization = _discretization((cells, cells), (1.0, 1.0), (True, True))
    law = phx.discretization.LinearSurfaceTensionLaw(0.05, slope, 0.5)
    material = two_phase_api.TwoPhaseMaterialPlan(
        liquid_density=1.0,
        gas_density=1.0,
        liquid_viscosity=viscosity,
        gas_viscosity=viscosity,
    )
    two_phase = two_phase_api.IncompressibleTwoPhaseVOFPlan(
        discretization,
        material,
        maximum_iterations=1000,
        surface_tension_law=law,
        surface_tension_scalar="temperature",
    ).prepare()
    centers = np.asarray(discretization.cell_centers)
    alpha = _disk_fraction((cells, cells), (1.0, 1.0), (0.5, 0.5), radius)
    alpha_array = jnp.asarray(alpha)
    temperature = jnp.asarray(gradient * centers[..., 0])
    geometry = two_phase.interface_geometry(alpha_array)
    if geometry.curvature is None:
        raise RuntimeError("Thermocapillary qualification requires curvature geometry.")
    policy = two_phase.capillarity.policy
    if not isinstance(policy, phx.discretization.VariableSurfaceTensionPolicy):
        raise RuntimeError("Thermocapillary qualification requires a variable law.")
    scalar_gradient = jnp.broadcast_to(
        jnp.asarray((gradient, 0.0), dtype=alpha_array.dtype),
        alpha_array.shape + (1, 2),
    )
    surface = policy.evaluate(
        discretization.cell_centers,
        temperature[..., None],
        geometry.plic.normal,
        scalar_gradient,
    )
    initial_force = two_phase.capillarity.evaluate(
        alpha_array,
        geometry.curvature.evidence,
        variable_surface_tension=surface,
    )
    net_marangoni_force = jnp.stack(
        tuple(
            jnp.sum(component * dual)
            for component, dual in zip(
                initial_force.marangoni_face_force,
                two_phase.operators.face_dual_measures,
                strict=True,
            )
        )
    )
    reference_force = float(slope * gradient * np.pi * radius)
    reference = -slope * gradient * radius / (4.0 * (viscosity + viscosity))
    force_balance_velocity = reference * (
        float(net_marangoni_force[0]) / reference_force
    )
    force_balance_relative_error = (
        abs(force_balance_velocity - reference) / abs(reference)
    )
    fluid = two_phase.initial_state(
        alpha_array,
        material_scalars={"temperature": temperature},
    )
    method = two_phase_api.IncompressibleTwoPhaseVOFMethod(two_phase)
    continuation = method.initial_continuation(fluid)
    initial_content = float(jnp.sum(fluid.material_scalar_content["temperature"]))
    step_size = 1.0e-3
    completed = 0
    maximum_steps = 12
    maximum_unsupported_faces = 0
    for index in range(maximum_steps):
        result = method.step(
            jnp.asarray(index, dtype=jnp.int32),
            jnp.asarray(index * step_size),
            continuation,
            jnp.asarray(step_size),
            None,
        )
        evidence = result.candidate_state.evidence
        if evidence is not None:
            maximum_unsupported_faces = max(
                maximum_unsupported_faces,
                int(evidence.unsupported_face_count),
            )
        if not bool(result.successful):
            break
        continuation = result.accepted_state
        completed += 1
    velocity = two_phase.velocity(continuation.state)[0]
    cell_velocity = 0.5 * (velocity + jnp.roll(velocity, -1, axis=0))
    measured = float(jnp.sum(cell_velocity * alpha) / jnp.sum(alpha))
    relative_error = abs(measured - reference) / abs(reference)
    final_content = float(
        jnp.sum(continuation.state.material_scalar_content["temperature"])
    )
    conservation = abs(final_content - initial_content) / max(abs(initial_content), 1.0)
    return {
        "cells": cells,
        "steps": completed,
        "measured_velocity": measured,
        "reference_velocity": reference,
        "relative_error": relative_error,
        "direction_agrees": measured * reference > 0.0,
        "material_scalar_conservation": conservation,
        "maximum_unsupported_faces": maximum_unsupported_faces,
        "completed": completed == maximum_steps,
        "net_marangoni_force": [float(value) for value in net_marangoni_force],
        "reference_marangoni_force": reference_force,
        "force_balance_velocity": force_balance_velocity,
        "force_balance_relative_error": force_balance_relative_error,
        "force_valid": bool(initial_force.valid),
    }


def thermocapillary() -> Record:
    """Bounded 2D refinement campaign against Young–Goldstein–Block."""

    refinements = tuple(_thermocapillary_resolution(cells) for cells in (16, 24))
    coarse, fine = refinements
    improved = (
        fine["force_balance_relative_error"]
        < coarse["force_balance_relative_error"]
    )
    passed = (
        all(
            record["completed"]
            and record["direction_agrees"]
            and record["material_scalar_conservation"] < 1.0e-12
            and record["force_valid"]
            and record["maximum_unsupported_faces"] == 0
            for record in refinements
        )
        and improved
        and fine["force_balance_relative_error"] < 0.1
    )
    return {
        "reference": "Young, Goldstein & Block 1959; 2D k'=k analogue",
        **fine,
        "refinements": list(refinements),
        "force_balance_error_improved": improved,
        "passed": passed,
    }

CAMPAIGNS: dict[str, Callable[[], Record]] = {
    "breathing-closed": breathing_closed,
    "breathing-vented": breathing_vented,
    "boyle-rise": boyle_rise,
    "thermocapillary": thermocapillary,
}


def run_qualification(names: tuple[str, ...], /) -> Record:
    campaigns = {}
    for name in names:
        started = time.perf_counter()
        record = CAMPAIGNS[name]()
        record["seconds"] = time.perf_counter() - started
        campaigns[name] = record
    return {
        "identity": _runtime_identity(),
        "campaigns": campaigns,
        "successful": all(bool(record["passed"]) for record in campaigns.values()),
    }


def _json_ready(value: object, /) -> object:
    """Record non-finite metrics explicitly instead of failing JSON encoding."""

    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--campaign",
        action="append",
        choices=tuple(CAMPAIGNS),
        help="Run only the named campaign(s); by default every campaign runs.",
    )
    arguments = parser.parse_args()
    report = run_qualification(tuple(arguments.campaign or CAMPAIGNS))
    encoded = json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
