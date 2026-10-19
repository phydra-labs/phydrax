#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Compile exact derivative and direct-layer routes before adoption.

Default capacities match the committed baseline: derivative dimensions 8/16/32/64
and direct layer (targets, sources) capacities (64,1024)/(128,4096)/(256,16384).
--smoke uses the first two capacities of each route; --full explicitly selects
the default campaign. Candidate gates match both capacity and width, never row
position. Baseline records are captured before the corresponding package changes.
Derivative candidates are exact AD prototypes; the layer candidate uses the
shipped explicit target policy. Both routes retain timing samples, including
target chunks 8/32/64; timing acceptance belongs to the quiet integration wave.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.geometry import Circle
from phydrax.operators.differential._requests import evaluate_fused_coordinate_derivatives
from phydrax.operators.integral.layer_potential import (
    BoundaryPanelization2D,
    LaplaceLayerPotential2D,
)
from phydrax.units import METER

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


class _Compiled(Protocol):
    compiled: jax.stages.Compiled

    def __call__(self, *arguments: Array | LaplaceLayerPotential2D) -> Array: ...


class _Lowered(Protocol):
    def compile(self) -> _Compiled: ...


@runtime_checkable
class _Entry(Protocol):
    def __call__(self, *arguments: Array | LaplaceLayerPotential2D) -> Array: ...

    def lower(self, *arguments: Array | LaplaceLayerPotential2D) -> _Lowered: ...


def _require_entry(function: Callable[..., Array]) -> _Entry:
    if not isinstance(function, _Entry):
        raise TypeError("Operator measurement requires a callable lowering entry.")
    return function


def _field(x: Array, matrix: Array) -> Array:
    return jnp.sin(matrix @ x) + x**3


def _div_baseline(x: Array, matrix: Array) -> Array:
    evaluated = evaluate_fused_coordinate_derivatives(
        lambda point: _field(point, matrix), x, first_axes=tuple(range(x.size))
    )
    total = jnp.zeros_like(x[0])
    for axis, column in enumerate(evaluated.first_derivatives):
        total = total + column[axis]
    return total


def _div_candidate(x: Array, matrix: Array) -> Array:
    def component(axis: Array) -> Array:
        direction = jnp.zeros_like(x).at[axis].set(1.0)
        return jax.jvp(lambda point: _field(point, matrix)[axis], (x,), (direction,))[1]

    values = jax.vmap(component)(jnp.arange(x.size))
    total = jnp.zeros_like(x[0])
    for axis in range(x.size):
        total = total + values[axis]
    return total


def _scalar(x: Array, matrix: Array) -> Array:
    return jnp.sum(jnp.sin(matrix @ x)) + jnp.sum(x**3)


def _diagonal_baseline(x: Array, matrix: Array) -> Array:
    coefficients = _field(x, matrix)
    gradient = jax.grad(_scalar)(x, matrix)
    coefficient_gradient = jax.jacrev(_field)(x, matrix)
    hessian = jax.jacrev(jax.grad(_scalar))(x, matrix)
    return jnp.sum(jnp.diag(coefficient_gradient) * gradient) + jnp.sum(
        coefficients * jnp.diag(hessian)
    )


def _diagonal_candidate(x: Array, matrix: Array) -> Array:
    coefficients = _field(x, matrix)

    def component(axis: Array) -> tuple[Array, Array]:
        direction = jnp.zeros_like(x).at[axis].set(1.0)

        def first(point: Array) -> Array:
            return jax.jvp(
                lambda current: _scalar(current, matrix), (point,), (direction,)
            )[1]

        du, d2u = jax.jvp(first, (x,), (direction,))
        dk = jax.jvp(lambda point: _field(point, matrix)[axis], (x,), (direction,))[1]
        return dk * du, coefficients[axis] * d2u

    first, second = jax.vmap(component)(jnp.arange(x.size))
    return jnp.sum(first) + jnp.sum(second)


def _drift_baseline(x: Array, matrix: Array) -> Array:
    return jax.jacrev(_field)(x, matrix) @ jnp.cos(x)


def _drift_candidate(x: Array, matrix: Array) -> Array:
    return jax.jvp(lambda point: _field(point, matrix), (x,), (jnp.cos(x),))[1]


def _layer_one(target: Array, sources: Array, density: Array) -> Array:
    # Independent direct single-layer source sum with the package reduction shape.
    radii = jnp.sqrt(jnp.sum((target[None, :] - sources) ** 2, axis=-1))
    return jnp.sum(-jnp.log(radii) * density / (2.0 * jnp.pi))


def _layer_baseline(targets: Array, potential: LaplaceLayerPotential2D) -> Array:
    return potential._evaluate_direct(targets)


def _layer_candidate(
    targets: Array, potential: LaplaceLayerPotential2D, *, chunk_size: int = 8
) -> Array:
    return potential._evaluate_direct(
        targets, target_execution="chunked", target_chunk_size=chunk_size
    )


def _measure(
    function: Callable[..., Array], arguments: tuple[Array | LaplaceLayerPotential2D, ...]
) -> tuple[Array, dict[str, Any]]:
    entry = _require_entry(eqx.filter_jit(function))
    (compiled, timing), host_peak = host_allocation_peak(
        lambda: measure_lower_and_compile(
            lambda: entry.lower(*arguments), lambda lowered: lowered.compile()
        )
    )
    result, warm = measure_repeated(lambda: compiled(*arguments), warmup=1, repeats=2)
    compiler = compiler_evidence(
        compiled.compiled.cost_analysis(),
        compiled.compiled.memory_analysis(),
        source="operator-workspace",
    )
    return result, {
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(compiler),
        "output_bytes": logical_array_bytes(result),
        "host_peak_bytes": host_peak,
    }


def _matching_baseline(
    series: list[dict[str, Any]], baseline: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Match the complete problem size without inventing historical measurements."""
    by_size = {(row["capacity"], row["width"]): row for row in baseline}
    if len(by_size) != len(baseline):
        raise ValueError("Baseline operator capacities and widths must be unique.")
    matched = []
    for row in series:
        size = (row["capacity"], row["width"])
        if size not in by_size:
            raise ValueError(f"Baseline has no operator case at capacity/width {size}.")
        matched.append(by_size[size])
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "candidate"), default="candidate")
    campaign = parser.add_mutually_exclusive_group()
    campaign.add_argument("--full", action="store_true")
    campaign.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    dimensions = (8, 16) if options.smoke else (8, 16, 32, 64)
    layer_sizes = (
        ((64, 1024), (128, 4096))
        if options.smoke
        else ((64, 1024), (128, 4096), (256, 16384))
    )
    baseline_path = Path(__file__).with_name("memory_operator_workspace_baseline.json")
    baseline = (
        json.loads(baseline_path.read_text()) if options.phase == "candidate" else None
    )
    routes: dict[str, tuple[Callable[..., Array], Callable[..., Array]]] = {
        "D10": (_div_baseline, _div_candidate),
        "D11": (_diagonal_baseline, _diagonal_candidate),
        "D12": (_drift_baseline, _drift_candidate),
        "D13": (_layer_baseline, _layer_candidate),
    }
    cases: dict[str, list[dict[str, Any]]] = {item: [] for item in routes}
    for item, (original, candidate) in routes.items():
        capacities = layer_sizes if item == "D13" else tuple((n, n) for n in dimensions)
        for capacity, width in capacities:
            if item == "D13":
                geometry = Circle((0.0, 0.0), 1.0).compile(coordinate_unit=METER)
                atlas = geometry.boundary_atlas
                owner_count = sum(bool(owner) for owner in np.asarray(atlas.seam_owner))
                panelization = BoundaryPanelization2D(
                    atlas,
                    panels_per_chart=width // (4 * owner_count),
                    quadrature_order=4,
                    geometry=geometry,
                )
                potential = LaplaceLayerPotential2D(
                    panelization,
                    kind="single",
                    density=jnp.linspace(0.5, 1.5, panelization.node_count),
                )
                targets = jnp.stack(
                    (jnp.linspace(-0.4, 0.4, capacity), jnp.full((capacity,), 0.2)),
                    axis=-1,
                )
                arguments = (targets, potential)
                logical = 8 * min(capacity, 8) * panelization.node_count
            else:
                arguments = (
                    jnp.linspace(-0.3, 0.4, capacity),
                    jnp.reshape(
                        jnp.linspace(-0.2, 0.3, capacity**2), (capacity, capacity)
                    ),
                )
                logical = 8 * capacity if item in ("D10", "D12") else 16 * capacity
            function = original if options.phase == "baseline" else candidate
            result, evidence = _measure(function, arguments)
            chunk_routes: list[dict[str, Any]] = []
            if options.phase == "candidate":
                reference = original(*arguments)
                np.testing.assert_allclose(
                    np.asarray(result), np.asarray(reference), rtol=2e-12, atol=2e-12
                )
                if item == "D13":
                    for chunk in (1, 3, 8, 32, 64, capacity + 1):
                        chunked = _layer_candidate(targets, potential, chunk_size=chunk)
                        np.testing.assert_allclose(
                            np.asarray(chunked),
                            np.asarray(reference),
                            rtol=2e-12,
                            atol=2e-12,
                        )
                    chunk_routes.append({"chunk_size": 8, **evidence})
                    for chunk in (32, 64):
                        tiled, tiled_evidence = _measure(
                            partial(_layer_candidate, chunk_size=chunk), arguments
                        )
                        np.testing.assert_allclose(
                            np.asarray(tiled),
                            np.asarray(reference),
                            rtol=2e-12,
                            atol=2e-12,
                        )
                        chunk_routes.append({"chunk_size": chunk, **tiled_evidence})
                    independent = jax.vmap(_layer_one, in_axes=(0, None, None))(
                        targets,
                        panelization.points,
                        potential.density * panelization.weights,
                    )
                    np.testing.assert_allclose(
                        np.asarray(result),
                        np.asarray(independent),
                        rtol=2e-12,
                        atol=2e-12,
                    )
            cases[item].append(
                {
                    "capacity": capacity,
                    "width": width,
                    "logical_candidate_workspace_bytes": logical,
                    "chunk_routes": chunk_routes,
                    "parity_chunk_sizes": [1, 3, 8, 32, 64, capacity + 1]
                    if item == "D13" and options.phase == "candidate"
                    else [],
                    **evidence,
                }
            )
            jax.clear_caches()
    gates: dict[str, Any] = {}
    if baseline is not None:
        for item, series in cases.items():
            capacities = [row["capacity"] for row in series]
            matched = _matching_baseline(series, baseline["cases"][item])
            previous = [row["compiler"]["temporary_bytes"] for row in matched]
            current = [row["compiler"]["temporary_bytes"] for row in series]
            gate = evaluate_memory_gate(
                capacities, previous, current, MemoryGatePolicy()
            ).to_dict()
            gate["logical_scaling_exponent"] = fitted_scaling_exponent(
                capacities, [row["logical_candidate_workspace_bytes"] for row in series]
            )
            gate["compiler_scaling_exponent"] = (
                fitted_scaling_exponent(capacities, current) if all(current) else None
            )
            gate["declared_formula"] = (
                "8*min(T,8)*S" if item == "D13" else ("16*d" if item == "D11" else "8*d")
            )
            expected_exponent = gate["logical_scaling_exponent"]
            actual_exponent = gate["compiler_scaling_exponent"]
            gate["scaling_matches_declared_formula"] = (
                actual_exponent is not None
                and abs(actual_exponent - expected_exponent) <= 0.25
            )
            gate["fraction_decision"] = gate["decision"]
            if not gate["scaling_matches_declared_formula"]:
                gate["decision"] = "retain-baseline"
            gates[item] = gate
    record = {
        "phase": options.phase,
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "gates": gates,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }
    record["identity"] = capture_benchmark_identity(
        Path.cwd(), Path(__file__).resolve(), tuple(record)
    ).to_dict()
    destination = options.output or (
        baseline_path
        if options.phase == "baseline"
        else Path(__file__).with_suffix(".json")
    )
    write_json_atomic(destination, record)
    print(json.dumps(gates or {"baseline": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
