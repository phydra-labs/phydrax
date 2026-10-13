"""Bounded contact, London, and PTV preparation/execution evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.applications.superconductivity import (
    ThinFilmLondonPlan,
    ThinFilmLondonResult,
)
from phydrax.geometry import TriangleMesh

from ._io import write_json_atomic
from ._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    host_allocation_peak,
    MemoryGatePolicy,
    process_peak_rss_bytes,
)
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
)


def _contact(keys: Array, valid: Array) -> tuple[Array, Array, Array, Array]:
    equality = (keys[:, None] == keys[None, :]) & valid[:, None] & valid[None, :]
    return (
        jnp.any(equality, axis=1),
        jnp.argmax(equality, axis=1),
        jnp.sum(equality, axis=1) > 1,
        jnp.any(equality, axis=0),
    )


def _ptv(pixels: Array, depth: Array, valid: Array) -> Array:
    delta = pixels[:, None, :] - pixels[None, :, :]
    return valid & ~jnp.any(
        valid[None, :]
        & (depth[None, :] < depth[:, None])
        & (jnp.sum(delta**2, axis=-1) <= 1.0),
        axis=1,
    )


def _dense_field(plan: ThinFilmLondonPlan, current: Array) -> Array:
    from phydrax._physical import ElectromagneticScaleContract

    centers = jnp.mean(plan.mesh.vertices[plan.mesh.faces], axis=1)
    separation = centers[:, None, :] - centers[None, :, :]
    radius = (jnp.sum(separation**2, axis=-1) + plan.softening_length**2) ** 1.5
    numerator = jnp.sum(
        plan.plane_normal * jnp.cross(current[None, :, :], separation), axis=-1
    )
    return (
        float(ElectromagneticScaleContract.si().vacuum_permeability)
        / (4.0 * jnp.pi)
        * jnp.sum(plan.operators.face_area[None, :] * numerator / radius, axis=1)
    )


def _mesh(count: int) -> TriangleMesh:
    theta = np.arange(count, dtype=np.float64) * (2 * np.pi / count)
    vertices = np.concatenate(
        (
            np.column_stack((np.cos(theta), np.sin(theta), np.zeros(count))),
            np.zeros((1, 3)),
        ),
        axis=0,
    )
    faces = np.column_stack(
        (np.arange(count), (np.arange(count) + 1) % count, np.full(count, count))
    ).astype(np.int32)
    return TriangleMesh(vertices, faces, source_id=f"memory-london-{count}")


@runtime_checkable
class _Entry(Protocol):
    def __call__(self, *args: Any) -> Any: ...
    def lower(self, *args: Any) -> Any: ...


def _measure(
    function: Any, args: tuple[Any, ...], reference: Any = None
) -> dict[str, Any]:
    entry = eqx.filter_jit(function)
    if not isinstance(entry, _Entry):
        raise TypeError("Benchmark requires a lowering entry.")
    executable, timing = measure_lower_and_compile(
        lambda: entry.lower(*args), lambda lowered: lowered.compile()
    )
    result, warm = measure_repeated(lambda: executable(*args), warmup=1, repeats=2)
    evidence = compiler_evidence(
        executable.compiled.cost_analysis(),
        executable.compiled.memory_analysis(),
        source="application-memory",
        unavailable_reason="Compiler estimate unavailable.",
    )
    scientific: dict[str, Any] = {}
    if isinstance(result, ThinFilmLondonResult):
        scientific = {
            "successful": bool(result.evidence.successful),
            "linear_residual": float(result.evidence.linear_residual),
            "constraint_residual": float(result.evidence.constraint_residual),
            "minimum_energy_curvature": float(result.evidence.minimum_energy_curvature),
        }
    elif reference is not None:
        actual = (
            result[0]
            if isinstance(reference, Array) and isinstance(result, tuple)
            else result
        )
        actual_leaves = eqx.filter(actual, eqx.is_array)
        reference_leaves = eqx.filter(reference, eqx.is_array)
        import jax

        defects = [
            float(
                np.max(
                    np.abs(
                        np.asarray(a).astype(np.float64)
                        - np.asarray(b).astype(np.float64)
                    ),
                    initial=0.0,
                )
            )
            for a, b in zip(
                jax.tree.leaves(actual_leaves),
                jax.tree.leaves(reference_leaves),
                strict=True,
            )
        ]
        scientific["reference_defect"] = max(defects, default=0.0)
        if isinstance(reference, Array) and isinstance(result, tuple):
            scientific["successful"] = bool(result[1].successful)
    return {
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(evidence),
        "retained_output_bytes": logical_array_bytes(result),
        **scientific,
    }


def _prepare_plan(
    factory: Callable[[], ThinFilmLondonPlan],
) -> tuple[ThinFilmLondonPlan, int, float]:
    (plan, seconds), peak = host_allocation_peak(lambda: measure_host(factory))
    return plan, peak, seconds


def _temporary_exponent(capacities: list[int], byte_counts: list[int]) -> float | None:
    # Fully fused smoke kernels have zero temporary bytes and no logarithmic fit.
    if 0 in byte_counts:
        return None
    return fitted_scaling_exponent(capacities, byte_counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--faces", type=int, nargs="+")
    options = parser.parse_args()
    cases: list[dict[str, Any]] = []
    sizes = options.sizes or ((32, 64) if options.smoke else (64, 256, 1024))
    for count in sizes:
        keys = jnp.arange(count, dtype=jnp.int64)
        valid = jnp.ones(count, dtype=jnp.bool_)
        pixels = jnp.column_stack(
            (jnp.arange(count, dtype=jnp.float64) * 2, jnp.zeros(count))
        )
        contact = _contact
        ptv = _ptv
        if not options.baseline:
            from phydrax.applications.contact._route_state import _route_matches
            from phydrax.velocimetry.synthetic import PTVScenarioKind, PTVScenarioPlan
            from phydrax.velocimetry.synthetic._ptv import _occlusion_camera

            contact = lambda k, v: _route_matches(k, v, k, v)
            ptv_plan = PTVScenarioPlan(
                PTVScenarioKind.OCCLUSION,
                particle_capacity=count,
                image_shape=(count * 3, count * 3),
                occlusion_radius=1.0,
            )
            ptv = lambda p, d, v: _occlusion_camera(ptv_plan, p, d, v)
        cases.append(
            {
                "area": "contact",
                "capacity": count,
                "logical_workspace_bytes": 64 * count,
                **_measure(contact, (keys, valid), _contact(keys, valid)),
            }
        )
        cases.append(
            {
                "area": "ptv",
                "capacity": count,
                "logical_workspace_bytes": 4 * count * min(count, 256) + 64 * count,
                **_measure(
                    ptv,
                    (pixels, jnp.arange(count, dtype=jnp.float64), valid),
                    _ptv(pixels, jnp.arange(count, dtype=jnp.float64), valid),
                ),
            }
        )
        if not options.baseline:
            crowded = _occlusion_camera(
                ptv_plan,
                jnp.zeros_like(pixels),
                jnp.arange(count, dtype=jnp.float64),
                valid,
            )[1]
            cases[-1]["crowded_successful"] = bool(crowded.successful)
            cases[-1]["crowded_overflow_rows"] = int(crowded.overflow_rows)
            cases[-1]["crowded_pair_overflow"] = bool(crowded.pair_overflow)
            cases[-1]["pair_capacity"] = ptv_plan.occlusion_relation.maximum_pairs
            cases[-1]["candidate_capacity"] = (
                ptv_plan.occlusion_relation.maximum_candidates
            )
    faces = options.faces or ((4, 8) if options.smoke else (20, 40, 80))
    matrix_free_cases: list[dict[str, Any]] = []
    field_cases: list[dict[str, Any]] = []
    for count in faces:
        mesh = _mesh(count)
        plan, peak, preparation_seconds = _prepare_plan(
            lambda: ThinFilmLondonPlan(mesh, pearl_length=0.2)
        )
        cases.append(
            {
                "area": "london",
                "capacity": count,
                "host_preparation_peak_bytes": peak,
                "preparation_seconds": preparation_seconds,
                "retained_plan_bytes": logical_array_bytes(plan),
                **_measure(
                    lambda p, b: p.solve(b), (plan, jnp.asarray(0.1, dtype=jnp.float64))
                ),
            }
        )
        if not options.baseline:
            from phydrax.applications.superconductivity import LondonSolvePolicy

            current = jnp.sin(jnp.arange(count * 3, dtype=jnp.float64)).reshape(
                (count, 3)
            )
            reference_field = _dense_field(plan, current)
            field_cases.append(
                {
                    "capacity": count,
                    "logical_tile_workspace_bytes": 8 * min(count, 8) * count * 8,
                    "dense": _measure(_dense_field, (plan, current), reference_field),
                    "tiled": _measure(
                        lambda p, c: p.energy_action.normal_field(c, p.plane_normal),
                        (plan, current),
                        reference_field,
                    ),
                }
            )
            for route in ("minres", "gmres"):
                from phydrax.applications.superconductivity import LondonSolveRoute
                from phydrax.typing import parse

                matrix_free, peak, preparation_seconds = _prepare_plan(
                    lambda: ThinFilmLondonPlan(
                        mesh,
                        pearl_length=0.2,
                        solve_policy=LondonSolvePolicy(
                            route=parse(route, LondonSolveRoute, "route")
                        ),
                    )
                )
                matrix_free_cases.append(
                    {
                        "route": route,
                        "capacity": count,
                        "logical_face_tile_bytes": 8 * min(count, 8) * count * 8,
                        "retained_plan_bytes": logical_array_bytes(matrix_free),
                        "host_preparation_peak_bytes": peak,
                        "preparation_seconds": preparation_seconds,
                        **_measure(
                            lambda p, b: p.solve(b),
                            (matrix_free, jnp.asarray(0.1, dtype=jnp.float64)),
                        ),
                    }
                )
    payload = {
        "environment": asdict(capture_environment()),
        "identity": asdict(
            capture_benchmark_identity(Path.cwd(), Path(__file__), ("cases",))
        ),
        "baseline": options.baseline,
        "cases": cases,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "matrix_free_london": matrix_free_cases,
        "field_kernel_reference_comparison": field_cases,
    }
    if not options.baseline:
        baseline_path = Path(__file__).with_name(
            "memory_contact_london_ptv_baseline.json"
        )
        baseline = json.loads(baseline_path.read_text())
        gates = {}
        for area in ("contact", "ptv", "london"):
            before = [case for case in baseline["cases"] if case["area"] == area]
            after = [case for case in cases if case["area"] == area]
            capacities = [case["capacity"] for case in after]
            if capacities == [case["capacity"] for case in before]:
                candidate_bytes = [case["compiler"]["temporary_bytes"] for case in after]
                gates[area] = {
                    **evaluate_memory_gate(
                        capacities,
                        [case["compiler"]["temporary_bytes"] for case in before],
                        candidate_bytes,
                        MemoryGatePolicy(),
                    ).to_dict(),
                    "fitted_temporary_exponent": _temporary_exponent(
                        capacities, candidate_bytes
                    ),
                }
        payload["gates"] = gates
        if len(field_cases) > 1:
            capacities = [case["capacity"] for case in field_cases]
            candidate_bytes = [
                case["tiled"]["compiler"]["temporary_bytes"] for case in field_cases
            ]
            payload["gates"]["london_field"] = {
                **evaluate_memory_gate(
                    capacities,
                    [
                        case["dense"]["compiler"]["temporary_bytes"]
                        for case in field_cases
                    ],
                    candidate_bytes,
                    MemoryGatePolicy(),
                ).to_dict(),
                "fitted_temporary_exponent": _temporary_exponent(
                    capacities, candidate_bytes
                ),
                "largest_two_temporary_exponent": _temporary_exponent(
                    capacities[-2:], candidate_bytes[-2:]
                ),
            }
        for route in ("minres", "gmres"):
            selected = [case for case in matrix_free_cases if case["route"] == route]
            if len(selected) > 1:
                payload[f"{route}_temporary_exponent"] = _temporary_exponent(
                    [case["capacity"] for case in selected],
                    [case["compiler"]["temporary_bytes"] for case in selected],
                )
    write_json_atomic(options.output, payload)


if __name__ == "__main__":
    main()
