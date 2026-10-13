"""G2 BRep compiler evidence for nearest cutover and retained winding candidates.

Run baseline before editing consumers, then candidate with the same box capacities.
Prepared topology and required outputs are recorded separately from compiler scratch.
The dynamic candidate includes refitting, not only an already-refitted query.
Nearest scan scratch follows Q*(k+log2(F)); dynamic refit is a separate prepared
topology phase, with combined compiler bytes still subject to the fraction gate.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, get_args, Literal, Protocol, runtime_checkable, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.geometry.simplicial import TriangleBVH, TriangleMesh
from phydrax.geometry.simplicial._mesh import _closest_points_on_triangles
from phydrax.typing import parse

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
    measure_lower_and_compile,
    measure_repeated,
)


EVIDENCE_FIELDS = ("environment", "identity", "route", "cases", "gates")


GeometryBRepBenchmarkRoute: TypeAlias = Literal["baseline", "candidate"]


def box_mesh(count: int) -> TriangleMesh:
    """Independent closed unit boxes with consistent outward face orientation."""
    corners = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    shifts = np.zeros((count, 1, 3), dtype=np.float64)
    shifts[:, 0, 0] = 2.0 * np.arange(count)
    return TriangleMesh(
        (corners[None] + shifts).reshape((-1, 3)),
        (faces[None] + 8 * np.arange(count, dtype=np.int32)[:, None, None]).reshape(
            (-1, 3)
        ),
    )


def dense_nearest(vertices: Array, faces: Array, points: Array) -> tuple[Array, ...]:
    """Existing fixed-topology nearest plus full second-distance sort."""
    triangles = vertices[faces]
    closest_by_face = jax.vmap(_closest_points_on_triangles, in_axes=(0, None))(
        points, triangles
    )
    squared = jnp.sum((closest_by_face - points[:, None]) ** 2, axis=-1)
    face = jnp.argmin(squared, axis=-1).astype(jnp.int32)
    closest = jnp.take_along_axis(closest_by_face, face[:, None, None], axis=1)[:, 0]
    triangle = triangles[face]
    normal = jnp.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
    second = jnp.sort(squared, axis=-1)[:, 1]
    return closest, normal, face, second


def candidate_nearest(tree: TriangleBVH, points: Array) -> tuple[Array, ...]:
    """Select IDs only through traversal; differentiate selected projections outside it."""
    selected = jax.lax.stop_gradient(tree.nearest_faces(points, k=2).items)
    triangles = tree.vertices[tree.faces[selected]]
    closest = jax.vmap(_closest_points_on_triangles)(points, triangles)
    second = jnp.sum((closest[:, 1] - points) ** 2, axis=-1)
    triangle = triangles[:, 0]
    normal = jnp.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
    return closest[:, 0], normal, selected[:, 0], second


def candidate_dynamic_nearest(
    tree: TriangleBVH, vertices: Array, points: Array
) -> tuple[Array, ...]:
    return candidate_nearest(tree.refit(vertices), points)


def dense_winding(vertices: Array, faces: Array, points: Array) -> Array:
    """Existing BRep solid-angle reduction, normalized for independent sign evidence."""
    triangles = vertices[faces]
    first = triangles[:, 0] - points[:, None]
    second = triangles[:, 1] - points[:, None]
    third = triangles[:, 2] - points[:, None]
    numerator = jnp.sum(first * jnp.cross(second, third), axis=-1)
    denominator = (
        jnp.linalg.norm(first, axis=-1)
        * jnp.linalg.norm(second, axis=-1)
        * jnp.linalg.norm(third, axis=-1)
        + jnp.sum(first * second, axis=-1) * jnp.linalg.norm(third, axis=-1)
        + jnp.sum(second * third, axis=-1) * jnp.linalg.norm(first, axis=-1)
        + jnp.sum(third * first, axis=-1) * jnp.linalg.norm(second, axis=-1)
    )
    return jnp.sum(2.0 * jnp.arctan2(numerator, denominator), axis=-1) / (4.0 * jnp.pi)


def candidate_winding(tree: TriangleBVH, points: Array) -> Array:
    return tree.winding_number(points).values


def candidate_dynamic_winding(tree: TriangleBVH, vertices: Array, points: Array) -> Array:
    return candidate_winding(tree.refit(vertices), points)


class _Compiled(Protocol):
    compiled: jax.stages.Compiled

    def __call__(self, *arguments: Any) -> Any: ...


class _Lowered(Protocol):
    def compile(self) -> _Compiled: ...


@runtime_checkable
class _Entry(Protocol):
    def lower(self, *arguments: Any) -> _Lowered: ...


def _measure(
    function: Callable[..., Any], arguments: tuple[Any, ...], repeats: int
) -> dict[str, Any]:
    entry = eqx.filter_jit(function)
    if not isinstance(entry, _Entry):
        raise TypeError("BRep memory measurement requires a lowering entry.")
    executable, timing = measure_lower_and_compile(
        lambda: entry.lower(*arguments), lambda lowered: lowered.compile()
    )
    result, warm = measure_repeated(
        lambda: executable(*arguments), warmup=1, repeats=repeats
    )
    evidence = compiler_evidence(
        executable.compiled.cost_analysis(),
        executable.compiled.memory_analysis(),
        source="jax-compiled-brep-query",
        unavailable_reason="Compiler did not supply all cost or memory estimates.",
    )
    return {
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(evidence),
        "output_bytes": logical_array_bytes(result),
        "argument_payload_bytes": logical_array_bytes(arguments),
    }


def _logarithmic_scan_fit(
    capacities: list[int], temporary_bytes: list[int], points: int
) -> dict[str, Any]:
    """Fit the prepared nearest scan's declared Q*(k+log F) scratch envelope."""
    logarithms = np.log2(np.asarray(capacities, dtype=np.float64))
    values = np.asarray(temporary_bytes, dtype=np.float64)
    design = np.column_stack((np.ones_like(logarithms), logarithms))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    predicted = design @ coefficients
    residuals = values - predicted
    normalized = values / (points * (2.0 + logarithms))
    normalized_ratios = normalized[1:] / normalized[:-1]
    return {
        "formula": "O(Q*(k+log2(F))), k=2; prepared topology/output excluded",
        "intercept_bytes": float(coefficients[0]),
        "bytes_per_log2_face": float(coefficients[1]),
        "predicted_bytes": predicted.tolist(),
        "residual_bytes": residuals.tolist(),
        "relative_residuals": (residuals / np.maximum(values, 1.0)).tolist(),
        "normalized_bytes": normalized.tolist(),
        "normalized_doubling_ratios": normalized_ratios.tolist(),
        "pass": len(capacities) >= 4
        and bool(np.all(normalized_ratios <= 1.25))
        and bool(np.all(np.abs(residuals) <= 0.25 * np.maximum(values, 1.0))),
    }


def _gates(baseline: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    capacities = [case["faces"] for case in cases]
    result: dict[str, Any] = {}
    for operation in ("nearest", "winding"):
        baseline_bytes = [
            case[operation]["compiler"]["temporary_bytes"] for case in baseline["cases"]
        ]
        for suffix in ("", "_dynamic"):
            name = operation + suffix
            candidate_bytes = [
                case[name]["compiler"]["temporary_bytes"] for case in cases
            ]
            gate = evaluate_memory_gate(
                capacities, baseline_bytes, candidate_bytes, MemoryGatePolicy()
            )
            ratios = [
                later / max(earlier, 1)
                for earlier, later in zip(candidate_bytes, candidate_bytes[1:])
            ]
            scaling_pass = all(ratio <= 1.25 for ratio in ratios)
            log_fit = (
                _logarithmic_scan_fit(
                    capacities,
                    [case["nearest"]["compiler"]["temporary_bytes"] for case in cases],
                    cases[0]["points"],
                )
                if operation == "nearest"
                else None
            )
            if log_fit is not None:
                scaling_pass = log_fit["pass"]
            base_time = baseline["cases"][0][operation]["warm"]["median_seconds"]
            candidate_time = cases[0][name]["warm"]["median_seconds"]
            result[name] = {
                "memory_gate": gate.to_dict(),
                "candidate_doubling_ratios": ratios,
                "candidate_scaling_exponent": fitted_scaling_exponent(
                    capacities, candidate_bytes
                ),
                "fixed_scan_axis_scaling_pass": scaling_pass,
                "logarithmic_scan_fit": log_fit,
                "phase": "refit-plus-query" if suffix else "prepared-query-scan",
                "scaling_scope": "prepared-query-scan; dynamic refit is a separate topology phase"
                if operation == "nearest"
                else "fixed-scan-axis",
                "smallest_dense_warm_seconds": base_time,
                "smallest_candidate_warm_seconds": candidate_time,
                "smallest_warm_ratio": candidate_time / base_time,
                "timing_acceptance": "deferred-serialized-wave",
                "decision": "adopt"
                if gate.decision == "adopt" and scaling_pass
                else "retain-baseline",
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route", choices=get_args(GeometryBRepBenchmarkRoute), required=True
    )
    parser.add_argument("--boxes", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--points", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).with_name("memory_geometry_brep_baseline.json"),
    )
    args = parser.parse_args()
    args.route = parse(args.route, GeometryBRepBenchmarkRoute, "route")
    if len(args.boxes) < 4 or any(b != 2 * a for a, b in zip(args.boxes, args.boxes[1:])):
        raise ValueError("Box capacities must double and contain at least four cases.")
    destination = args.output or Path(__file__).with_name(
        "memory_geometry_brep_baseline.json"
        if args.route == "baseline"
        else "memory_geometry_brep.json"
    )
    cases = []
    for count in args.boxes:
        mesh = box_mesh(count)
        points = jnp.asarray(
            np.random.default_rng(72).uniform(-0.25, 1.25, (args.points, 3)),
            dtype=jnp.float64,
        )
        case: dict[str, Any] = {
            "boxes": count,
            "faces": mesh.faces.shape[0],
            "vertices": mesh.vertices.shape[0],
            "points": args.points,
        }
        if args.route == "baseline":
            case["nearest"] = _measure(
                dense_nearest, (mesh.vertices, mesh.faces, points), args.repeats
            )
            case["winding"] = _measure(
                dense_winding, (mesh.vertices, mesh.faces, points), args.repeats
            )
            case["logical_pair_workspace_bytes"] = (
                args.points * mesh.faces.shape[0] * 3 * 8
            )
        else:
            tree, peak = host_allocation_peak(lambda: TriangleBVH(mesh))
            case["prepared_topology_bytes"] = logical_array_bytes(tree)
            case["host_preparation_peak_bytes"] = peak
            case["nearest"] = _measure(candidate_nearest, (tree, points), args.repeats)
            case["nearest_dynamic"] = _measure(
                candidate_dynamic_nearest, (tree, mesh.vertices, points), args.repeats
            )
            case["winding"] = _measure(candidate_winding, (tree, points), args.repeats)
            case["winding_dynamic"] = _measure(
                candidate_dynamic_winding, (tree, mesh.vertices, points), args.repeats
            )
        cases.append(case)
        jax.clear_caches()
    record = {
        "identity": capture_benchmark_identity(
            Path(__file__).parents[1], Path(__file__), EVIDENCE_FIELDS
        ).to_dict(),
        "environment": capture_environment().to_dict(),
        "route": args.route,
        "configuration": {
            "boxes": args.boxes,
            "points": args.points,
            "repeats": args.repeats,
        },
        "cases": cases,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "gates": {}
        if args.route == "baseline"
        else _gates(json.loads(args.baseline.read_text()), cases),
    }
    write_json_atomic(destination, record)
    print(
        json.dumps(
            {
                "output": str(destination),
                "gates": record["gates"],
                "peak_rss_bytes": record["process_peak_rss_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
