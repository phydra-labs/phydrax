#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Hysing rising-bubble qualification of structured two-phase VOF.

Reference: S. Hysing, S. Turek, D. Kuzmin, N. Parolini, E. Burman,
S. Ganesan and L. Tobiska, "Quantitative benchmark computations of
two-dimensional bubble dynamics", Int. J. Numer. Meth. Fluids 60 (2009)
1259-1288, doi:10.1002/fld.1934.  Problem definition: Section 2 and Table I
(domain [0, 1] x [0, 2], bubble radius 0.25 at (0.5, 0.5), no-slip top and
bottom, free-slip sides, final time 3).  Reference values: Tables XII (test
case 1) and XVI (test case 2), finest grids of the three groups (TP2D,
FreeLIFE, MooNMD).  Test case 2 break-up makes its circularity
non-comparable; the paper reports no agreement for it and it is not gated.

The campaign runs each case on several resolutions and time refinements,
feeds the simulated fields to
:func:`phydrax.applications.free_boundary.hysing_bubble_benchmark` and records
circularity, centroid, mean rise velocity, phase volume, projection residuals,
the energy ledger and step status.  Results are honest: a failed step is
reported with its evidence, not skipped.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization.finite_volume import CurvatureStatus
from phydrax.qualification import QualificationRuntimeIdentity


SOURCE = {
    "citation": (
        "Hysing et al., Int. J. Numer. Meth. Fluids 60 (2009) 1259-1288, "
        "doi:10.1002/fld.1934"
    ),
    "tables": "Table I (parameters), Table XII (case 1), Table XVI (case 2)",
}

CASES: dict[int, dict[str, float]] = {
    1: {
        "liquid_density": 1000.0,
        "gas_density": 100.0,
        "liquid_viscosity": 10.0,
        "gas_viscosity": 1.0,
        "gravity": 0.98,
        "surface_tension": 24.5,
    },
    2: {
        "liquid_density": 1000.0,
        "gas_density": 1.0,
        "liquid_viscosity": 10.0,
        "gas_viscosity": 0.1,
        "gravity": 0.98,
        "surface_tension": 1.96,
    },
}

# Finest-grid values of groups 1-3 (TP2D, FreeLIFE, MooNMD) and the gate width.
REFERENCES: dict[int, dict[str, dict[str, Any]]] = {
    1: {
        "minimum_circularity": {"groups": [0.9013, 0.9011, 0.9013], "tolerance": 5.0e-4},
        "minimum_circularity_time": {
            "groups": [1.9041, 1.8750, 1.9000],
            "tolerance": 0.05,
        },
        "maximum_rise_velocity": {
            "groups": [0.2417, 0.2421, 0.2417],
            "tolerance": 5.0e-4,
        },
        "maximum_rise_velocity_time": {
            "groups": [0.9213, 0.9313, 0.9239],
            "tolerance": 0.05,
        },
        "final_centroid": {"groups": [1.0813, 1.0799, 1.0817], "tolerance": 1.0e-3},
    },
    2: {
        "maximum_rise_velocity": {
            "groups": [0.2524, 0.2514, 0.2502],
            "tolerance": 2.0e-3,
        },
        "maximum_rise_velocity_time": {
            "groups": [0.7332, 0.7281, 0.7317],
            "tolerance": 0.05,
        },
        "second_maximum_rise_velocity": {
            "groups": [0.2434, 0.2440, 0.2393],
            "tolerance": 2.0e-3,
        },
        "final_centroid": {"groups": [1.1380, 1.1249, 1.1376], "tolerance": 3.0e-3},
    },
}


@dataclass(frozen=True)
class HysingRun:
    case: int
    cells_per_unit: int
    time_refinement: int
    step_size: float
    steps_requested: int
    steps_accepted: int
    completed: bool
    status: str
    failure: dict[str, Any] | None
    samples: list[dict[str, float]]
    metrics: dict[str, float | None]
    residuals: dict[str, float]
    ledger: dict[str, float]
    wall_seconds: float
    compile_seconds: float


def _runtime_identity() -> dict[str, Any]:
    build_id = canonical_fingerprint(
        {
            "kind": "two-phase-hysing-qualification-build",
            "phydrax": importlib.metadata.version("phydrax"),
            "phydrax_path": str(Path(phx.__file__).resolve().parent),
        }
    )
    environment_id = canonical_fingerprint(
        {
            "kind": "two-phase-hysing-qualification-environment",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
        }
    )
    identity = QualificationRuntimeIdentity(
        build_id,
        environment_id,
        jax.default_backend(),
        f"processes-{jax.process_count()}-devices-{jax.device_count()}",
        str(jnp.asarray(0.0).dtype),
    )
    return dict(identity.to_record())


def _bubble_liquid_fraction(count_x: int, count_y: int, width: float, /) -> np.ndarray:
    """Liquid fraction outside the r = 0.25 bubble at (0.5, 0.5).

    Exact chord lengths along x, composite Gauss quadrature along y.
    """

    nodes, weights = np.polynomial.legendre.leggauss(8)
    sub = 16
    points = ((np.arange(sub)[:, None] + 0.5 * (nodes[None, :] + 1.0)) / sub).ravel()
    point_weights = np.tile(0.5 * weights, sub) / sub
    y = (np.arange(count_y)[:, None] + points[None, :]) * width
    squared = 0.25**2 - (y - 0.5) ** 2
    half = np.sqrt(np.maximum(squared, 0.0))
    x_lower = (np.arange(count_x) * width)[:, None, None]
    chord = np.clip(
        np.minimum(0.5 + half, x_lower + width) - np.maximum(0.5 - half, x_lower),
        0.0,
        None,
    ) * (squared > 0.0)
    gas = np.clip((chord @ point_weights) / width, 0.0, 1.0)
    return 1.0 - gas


def _largest_contour(liquid: np.ndarray, centers: np.ndarray, /) -> np.ndarray:
    """Ordered iso-line ``liquid = 1/2`` enclosing the largest gas area.

    Marching squares on the cell-center dual grid; saddles are split by the
    square-center average so every edge crossing joins exactly two segments.
    """

    level = liquid - 0.5
    count_x, count_y = level.shape

    def crossing(key: tuple[str, int, int]) -> np.ndarray:
        kind, i, j = key
        a = (i, j)
        b = (i + 1, j) if kind == "h" else (i, j + 1)
        t = level[a] / (level[a] - level[b])
        return centers[a] + t * (centers[b] - centers[a])

    neighbors: dict[tuple[str, int, int], list[tuple[str, int, int]]] = {}

    def link(first: tuple[str, int, int], second: tuple[str, int, int]) -> None:
        neighbors.setdefault(first, []).append(second)
        neighbors.setdefault(second, []).append(first)

    for i in range(count_x - 1):
        for j in range(count_y - 1):
            corners = (level[i, j], level[i + 1, j], level[i + 1, j + 1], level[i, j + 1])
            edges = (("h", i, j), ("v", i + 1, j), ("h", i, j + 1), ("v", i, j))
            cut = [
                edges[k]
                for k in range(4)
                if (corners[k] < 0.0) != (corners[(k + 1) % 4] < 0.0)
            ]
            if len(cut) == 2:
                link(cut[0], cut[1])
            elif len(cut) == 4:
                center = 0.25 * sum(corners)
                if (center < 0.0) == (corners[0] < 0.0):
                    link(edges[0], edges[1])
                    link(edges[2], edges[3])
                else:
                    link(edges[0], edges[3])
                    link(edges[1], edges[2])
    loops = []
    visited: set[tuple[str, int, int]] = set()
    for start in neighbors:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        previous, current = start, neighbors[start][0]
        while current != start and current not in visited:
            loop.append(current)
            visited.add(current)
            following = [key for key in neighbors[current] if key != previous]
            if not following:
                break
            previous, current = current, following[0]
        points = np.asarray([crossing(key) for key in loop])
        x, y = points[:, 0], points[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        loops.append((area, points))
    return max(loops, key=lambda item: item[0])[1]


def _prepared(case: int, cells_per_unit: int, /) -> Any:
    parameters = CASES[case]
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(cells_per_unit),
            phx.discretization.UniformCellAxisSpec(2 * cells_per_unit),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 2.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    material = phx.applications.two_phase_flow.TwoPhaseMaterialPlan(
        liquid_density=parameters["liquid_density"],
        gas_density=parameters["gas_density"],
        liquid_viscosity=parameters["liquid_viscosity"],
        gas_viscosity=parameters["gas_viscosity"],
        surface_tension=parameters["surface_tension"],
    )
    sides = (
        phx.discretization.MACBoundarySide("x", "lower", "free-slip"),
        phx.discretization.MACBoundarySide("x", "upper", "free-slip"),
        phx.discretization.MACBoundarySide("y", "lower", "no-slip"),
        phx.discretization.MACBoundarySide("y", "upper", "no-slip"),
    )
    return discretization, phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan(
        discretization,
        material,
        tolerance=1.0e-9,
        maximum_iterations=4000,
        gravity=(0.0, -parameters["gravity"]),
        hydrostatic_reference=(0.5, 2.0),
        wall_sides=sides,
    ).prepare()


def _sample(
    discretization: Any, two_phase: Any, state: Any, time: float, /
) -> dict[str, float]:
    raw = np.asarray(two_phase.alpha(state))
    tolerance = phx.applications.two_phase_flow.alpha_bound_tolerance(raw.dtype)
    if raw.min() < -tolerance or raw.max() > 1.0 + tolerance:
        raise ValueError("Accepted alpha left the admitted rounding band.")
    # Observable only: accepted states may carry the admitted rounding
    # excursion, which the benchmark's [0, 1] fraction contract excludes.
    liquid = np.clip(raw, 0.0, 1.0)
    centers = np.asarray(discretization.cell_centers)
    velocity = two_phase.velocity(state)
    vertical = np.asarray(0.5 * (velocity[1][:, :-1] + velocity[1][:, 1:]))
    report = phx.applications.free_boundary.hysing_bubble_benchmark(
        (1.0 - liquid).reshape((-1,)),
        centers.reshape((-1, 2)),
        np.asarray(discretization.cell_volumes).reshape((-1,)),
        vertical.reshape((-1,)),
        _largest_contour(liquid, centers),
    )
    return {
        "time": time,
        "circularity": float(report.circularity),
        "centroid_y": float(np.asarray(report.centroid)[1]),
        "rise_velocity": float(report.mean_rise_velocity),
        "gas_area": float(report.area),
    }


def _finite_scalar(value: Any, /) -> float | None:
    scalar = float(value)
    return scalar if np.isfinite(scalar) else None


def _curvature_failure_evidence(two_phase: Any, state: Any, /) -> dict[str, Any]:
    """Re-evaluate a refused candidate and retain bounded-fit failure evidence."""

    geometry = two_phase.interface_geometry(two_phase.alpha(state))
    curvature = geometry.curvature
    policy = two_phase.curvature_plan
    if curvature is None or policy is None:
        return {"available": False}
    status = np.asarray(curvature.evidence.status)
    underresolved = status == int(CurvatureStatus.UNDERRESOLVED)
    support = np.asarray(curvature.fallback_support_count)
    rank = np.asarray(curvature.fallback_rank)
    condition = np.asarray(curvature.fallback_condition)
    residual = np.asarray(curvature.fallback_residual)
    candidate = np.asarray(curvature.fallback_candidate_curvature)
    attempted = np.asarray(curvature.fallback_attempted)
    parameter_count = 3 if policy.dimension == 2 else 6
    residual_limit = policy.fallback_residual_limit * min(policy.spacing)
    curvature_limit = policy.curvature_bound / min(policy.spacing)
    cells = []
    for index_array in np.argwhere(underresolved):
        index = tuple(int(value) for value in index_array)
        local_condition = _finite_scalar(condition[index])
        local_residual = _finite_scalar(residual[index])
        local_candidate = _finite_scalar(candidate[index])
        cells.append(
            {
                "index": list(index),
                "support": int(support[index]),
                "rank": int(rank[index]),
                "condition": local_condition,
                "residual": local_residual,
                "candidate_curvature": local_candidate,
                "fallback_attempted": bool(attempted[index]),
                "insufficient_support": int(support[index]) < parameter_count,
                "rank_deficient": int(rank[index]) < parameter_count,
                "condition_exceeded": (
                    local_condition is None
                    or local_condition > policy.fallback_condition_limit
                ),
                "residual_exceeded": (
                    local_residual is None or local_residual > residual_limit
                ),
                "curvature_bound_exceeded": (
                    local_candidate is not None and abs(local_candidate) > curvature_limit
                ),
                "bounded_stencil_exhausted": bool(attempted[index]),
                "missing_primary_plic": not bool(attempted[index]),
            }
        )
    return {
        "available": True,
        "sample_source": "primary-valid-plic-facets",
        "radius": policy.fallback_radius,
        "capacity": policy.fallback_capacity,
        "normal_alignment": policy.fallback_normal_alignment,
        "condition_limit": policy.fallback_condition_limit,
        "residual_limit": residual_limit,
        "curvature_limit": curvature_limit,
        "attempted_count": int(np.count_nonzero(curvature.fallback_attempted)),
        "fallback_count": int(curvature.fallback_count),
        "underresolved_count": int(curvature.underresolved_count),
        "underresolved_cells": cells,
    }


def _metrics(case: int, samples: list[dict[str, float]], /) -> dict[str, float | None]:
    times = np.asarray([sample["time"] for sample in samples])
    circularity = np.asarray([sample["circularity"] for sample in samples])
    rise = np.asarray([sample["rise_velocity"] for sample in samples])
    metrics: dict[str, float | None] = {
        "final_time": float(times[-1]),
        "final_centroid": float(samples[-1]["centroid_y"]),
        "gas_area_drift": float(samples[-1]["gas_area"] - samples[0]["gas_area"]),
    }
    if case == 1:
        index = int(np.argmin(circularity))
        metrics["minimum_circularity"] = float(circularity[index])
        metrics["minimum_circularity_time"] = float(times[index])
        first = int(np.argmax(rise))
        metrics["maximum_rise_velocity"] = float(rise[first])
        metrics["maximum_rise_velocity_time"] = float(times[first])
    else:
        early = times < 1.3
        first = int(np.argmax(np.where(early, rise, -np.inf)))
        metrics["maximum_rise_velocity"] = float(rise[first])
        metrics["maximum_rise_velocity_time"] = float(times[first])
        late = times >= 1.3
        metrics["second_maximum_rise_velocity"] = (
            float(np.max(rise[late])) if bool(np.any(late)) else None
        )
        metrics["minimum_circularity"] = float(np.min(circularity))
    return metrics


def run_case(
    case: int,
    cells_per_unit: int,
    time_refinement: int,
    final_time: float,
    sample_interval: float,
    /,
) -> HysingRun:
    parameters = CASES[case]
    discretization, two_phase = _prepared(case, cells_per_unit)
    width = 1.0 / cells_per_unit
    liquid = _bubble_liquid_fraction(cells_per_unit, 2 * cells_per_unit, width)
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(two_phase)
    continuation = method.initial_continuation(
        two_phase.initial_state(jnp.asarray(liquid))
    )
    mean_density = 0.5 * (parameters["liquid_density"] + parameters["gas_density"])
    capillary = float(
        np.sqrt(mean_density * width**3 / (2.0 * np.pi * parameters["surface_tension"]))
    )
    # A gravity-wave velocity scale bounds the local interfacial speed without
    # using a measured benchmark observable. This keeps the declared directional
    # Courant gate meaningful on every spatial row, including coarse TC2.
    gravity_velocity = float(np.sqrt(parameters["gravity"] * 2.0))
    base = min(0.8 * capillary, 0.25 * width / gravity_velocity)
    steps = int(np.ceil(final_time / base)) * time_refinement
    step_size = final_time / steps
    stride = max(1, int(round(sample_interval / step_size)))
    step = eqx.filter_jit(method.step)
    samples = [_sample(discretization, two_phase, continuation.state, 0.0)]
    started = perf_counter()
    compile_seconds = 0.0
    failure = None
    accepted = 0
    maximum_divergence = 0.0
    maximum_pressure_residual = 0.0
    maximum_balance = 0.0
    for index in range(steps):
        call_started = perf_counter()
        result = step(
            jnp.asarray(index, dtype=jnp.int32),
            jnp.asarray(index * step_size),
            continuation,
            jnp.asarray(step_size),
            None,
        )
        if index == 0:
            jax.block_until_ready(result.successful)
            compile_seconds = perf_counter() - call_started
        evidence = result.candidate_state.evidence
        if not bool(result.successful):
            failure = {
                "step": index,
                "time": index * step_size,
                "evidence": {
                    "alpha_minimum": float(evidence.alpha_minimum),
                    "alpha_maximum": float(evidence.alpha_maximum),
                    "advective_courant": float(evidence.advective_courant),
                    "capillary_step_limit": float(evidence.capillary_step_limit),
                    "curvature_underresolved_count": int(
                        evidence.curvature_underresolved_count
                    ),
                    "unsupported_face_count": int(evidence.unsupported_face_count),
                    "plic_residual": float(evidence.plic_residual),
                    "viscous_converged": bool(evidence.viscous_converged),
                    "divergence_residual": float(evidence.divergence_residual),
                    "finite": bool(evidence.finite),
                },
                "topology_valid": bool(result.candidate_state.topology.valid),
            }
            failure["curvature_fallback"] = _curvature_failure_evidence(
                two_phase,
                result.candidate_state.state,
            )
            break
        continuation = result.accepted_state
        accepted += 1
        maximum_divergence = max(maximum_divergence, float(evidence.divergence_residual))
        maximum_pressure_residual = max(
            maximum_pressure_residual, float(evidence.pressure_residual)
        )
        maximum_balance = max(maximum_balance, float(evidence.capillary_balance_residual))
        if (index + 1) % stride == 0 or index + 1 == steps:
            samples.append(
                _sample(
                    discretization, two_phase, continuation.state, (index + 1) * step_size
                )
            )
    ledger = continuation.ledger
    return HysingRun(
        case=case,
        cells_per_unit=cells_per_unit,
        time_refinement=time_refinement,
        step_size=step_size,
        steps_requested=steps,
        steps_accepted=accepted,
        completed=failure is None and accepted == steps,
        status="COMPLETED" if failure is None and accepted == steps else "STEP_FAILED",
        failure=failure,
        samples=samples,
        metrics=_metrics(case, samples),
        residuals={
            "maximum_divergence": maximum_divergence,
            "maximum_pressure_residual": maximum_pressure_residual,
            "maximum_capillary_balance": maximum_balance,
            "liquid_volume_change": float(ledger.liquid_volume_change),
        },
        ledger={
            "kinetic_energy_change": float(ledger.kinetic_energy_change),
            "gravitational_energy_change": float(ledger.gravitational_energy_change),
            "surface_energy_change": float(ledger.surface_energy_change),
            "viscous_dissipation": float(ledger.viscous_dissipation),
            "wall_work": float(ledger.wall_work),
            "capillary_work": float(ledger.capillary_work),
            "gravity_work": float(ledger.gravity_work),
            "work_energy_residual": float(ledger.work_energy_residual),
            "gravitational_energy_residual": float(ledger.gravitational_energy_residual),
            "surface_energy_residual": float(ledger.surface_energy_residual),
        },
        wall_seconds=perf_counter() - started,
        compile_seconds=compile_seconds,
    )


def _assessment(case: int, runs: list[HysingRun], /) -> dict[str, Any]:
    all_runs_completed = bool(runs) and all(
        run.completed
        and run.status == "COMPLETED"
        and run.steps_accepted == run.steps_requested
        for run in runs
    )
    if not runs:
        return {
            "finest": None,
            "gates": {},
            "all_runs_completed": False,
            "passed": False,
        }
    finest = max(runs, key=lambda run: (run.cells_per_unit, run.time_refinement))
    gates = {}
    for name, reference in REFERENCES[case].items():
        value = finest.metrics.get(name) if finest.completed else None
        low = min(reference["groups"]) - reference["tolerance"]
        high = max(reference["groups"]) + reference["tolerance"]
        gates[name] = {
            "value": value,
            "window": [low, high],
            "passed": value is not None and low <= value <= high,
        }
    trends = {}
    for name in REFERENCES[case]:
        by_resolution = sorted(
            (
                (run.cells_per_unit, run.metrics.get(name))
                for run in runs
                if run.completed and run.time_refinement == 1
            ),
            key=lambda item: item[0],
        )
        values = [value for _, value in by_resolution if value is not None]
        differences = [abs(b - a) for a, b in zip(values, values[1:], strict=False)]
        trends[name] = {
            "values": values,
            "successive_differences": differences,
            "contracting": len(differences) >= 2 and differences[-1] < differences[0],
        }
    return {
        "finest": {
            "cells_per_unit": finest.cells_per_unit,
            "time_refinement": finest.time_refinement,
            "status": finest.status,
        },
        "gates": gates,
        "trends": trends,
        "all_runs_completed": all_runs_completed,
        "passed": all_runs_completed and all(gate["passed"] for gate in gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", type=int, choices=(1, 2), action="append")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[20, 40, 80])
    parser.add_argument("--time-refinements", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--final-time", type=float, default=3.0)
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    cases = arguments.case or [1, 2]
    report: dict[str, Any] = {
        "source": SOURCE,
        "references": REFERENCES,
        "runtime": _runtime_identity(),
        "cases": {},
    }
    for case in cases:
        runs = []
        for resolution in arguments.resolutions:
            for refinement in arguments.time_refinements:
                run = run_case(
                    case,
                    resolution,
                    refinement,
                    arguments.final_time,
                    arguments.sample_interval,
                )
                runs.append(run)
                print(
                    json.dumps(
                        {
                            "case": case,
                            "cells_per_unit": resolution,
                            "time_refinement": refinement,
                            "completed": run.completed,
                            "status": run.status,
                            "metrics": run.metrics,
                            "wall_seconds": run.wall_seconds,
                        }
                    ),
                    flush=True,
                )
        report["cases"][str(case)] = {
            "runs": [asdict(run) for run in runs],
            "assessment": _assessment(case, runs),
        }
    report["successful"] = all(
        bool(record["assessment"]["passed"]) for record in report["cases"].values()
    )
    payload = json.dumps(report, indent=2, allow_nan=False)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n")
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
