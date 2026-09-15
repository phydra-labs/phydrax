#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark exact Kerr geometry and its branch-local coordinate derivatives."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)

import phydrax as phx


def _compiler_record(compiled) -> dict[str, object]:
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="The selected JAX backend did not report compiler analysis.",
    )
    record = asdict(evidence)
    record["estimated_device_memory_bytes"] = evidence.estimated_device_memory_bytes
    return record


def _measure(function, arguments, warmup, repeats):
    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(function).lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(*arguments), warmup=warmup, repeats=repeats
    )
    return result, {
        "compilation": asdict(compilation),
        "execution": execution.to_seconds_dict(),
        "compiler": _compiler_record(compiled),
    }


def _setup(point_count: int):
    chart = phx.metrix.CoordinateChart(
        "benchmark-kerr-boyer-lindquist", ("t", "r", "theta", "phi")
    )
    mass = jnp.asarray(1.0)
    spin = jnp.asarray(0.6)
    horizon_radii = phx.metrix.kerr_horizon_radii(mass, spin)
    fraction = (jnp.arange(point_count, dtype=mass.dtype) + 0.5) / point_count
    points = jnp.stack(
        (
            0.25 * fraction,
            horizon_radii[1] + 0.25 + 18.0 * fraction,
            0.2 + (jnp.pi - 0.4) * fraction,
            -jnp.pi + 2.0 * jnp.pi * fraction,
        ),
        axis=-1,
    )
    direction = jnp.stack(
        (
            jnp.zeros_like(fraction),
            0.01 * jnp.ones_like(fraction),
            0.005 * jnp.cos(2.0 * jnp.pi * fraction),
            jnp.zeros_like(fraction),
        ),
        axis=-1,
    )
    metric = phx.metrix.kerr_boyer_lindquist_metric(
        mass, spin, chart=chart
    )
    return chart, metric, mass, spin, horizon_radii, points, direction


def run(point_count: int, warmup: int, repeats: int) -> dict[str, object]:
    environment = capture_environment().to_dict()
    setup, setup_seconds = measure_synchronized(lambda: _setup(point_count))
    chart, metric, mass, spin, horizon_radii, points, direction = setup

    def geometry_kernel(sample_points):
        matrices = jax.vmap(metric)(sample_points)
        domain = phx.metrix.kerr_boyer_lindquist_domain_evidence(
            mass, spin, sample_points, chart=chart
        )
        invariants = jax.vmap(
            lambda point: (
                phx.metrix.kerr_kretschmann_scalar(
                    mass, spin, point[1], point[2]
                ),
                phx.metrix.kerr_pontryagin_scalar(
                    mass, spin, point[1], point[2]
                ),
            )
        )(sample_points)
        determinants = jax.vmap(jnp.linalg.det)(matrices)
        return (
            matrices,
            determinants,
            invariants[0],
            invariants[1],
            domain.valid,
            domain.finite,
            domain.physically_valid,
            domain.derivative_valid,
            domain.status,
        )

    def differential_kernel(sample_points, tangent_points):
        def observables(value):
            matrices = jax.vmap(metric)(value)
            curvature = jax.vmap(
                lambda point: phx.metrix.kerr_kretschmann_scalar(
                    mass, spin, point[1], point[2]
                )
            )(value)
            dual_curvature = jax.vmap(
                lambda point: phx.metrix.kerr_pontryagin_scalar(
                    mass, spin, point[1], point[2]
                )
            )(value)
            return matrices, curvature, dual_curvature

        _, tangents = jax.jvp(
            observables, (sample_points,), (tangent_points,)
        )
        return tangents

    primal, primal_performance = _measure(
        geometry_kernel, (points,), warmup, repeats
    )
    tangents, derivative_performance = _measure(
        differential_kernel, (points, direction), warmup, repeats
    )
    matrices, determinants = primal[0], primal[1]
    sigma = points[:, 1] ** 2 + spin**2 * jnp.cos(points[:, 2]) ** 2
    expected_determinant = -(sigma * jnp.sin(points[:, 2])) ** 2
    determinant_relative_error = jnp.max(
        jnp.abs(determinants - expected_determinant)
        / jnp.maximum(jnp.abs(expected_determinant), jnp.finfo(points.dtype).tiny)
    )
    symmetry_residual = jnp.max(
        jnp.abs(matrices - jnp.swapaxes(matrices, -1, -2))
    )
    invariant_tolerance = 1_000.0 * jnp.finfo(points.dtype).eps
    curvature_invariants_finite = bool(
        jnp.all(jnp.isfinite(primal[2])) & jnp.all(jnp.isfinite(primal[3]))
    )
    tangent_finite = all(bool(jnp.all(jnp.isfinite(value))) for value in tangents)
    status_values, status_counts = np.unique(
        np.asarray(primal[8]), return_counts=True
    )
    mean_seconds = primal_performance["execution"]["mean_seconds"]
    successful = bool(
        jnp.all(primal[4])
        & jnp.all(primal[5])
        & jnp.all(primal[6])
        & jnp.all(primal[7])
        & curvature_invariants_finite
        & (symmetry_residual <= invariant_tolerance)
        & (determinant_relative_error <= invariant_tolerance)
    ) and tangent_finite
    return {
        "identities": {
            "benchmark": "black-hole-geometry",
            "kernel": "exact-kerr-boyer-lindquist-metric-and-invariants",
            "chart": chart.name,
            "coordinate_order": list(chart.coordinates),
            "convention": metric.convention,
        },
        "configuration": {
            "point_capacity": point_count,
            "mass": float(mass),
            "spin": float(spin),
            "outer_horizon_radius": float(horizon_radii[1]),
            "invariant_tolerance": float(invariant_tolerance),
            "warmup": warmup,
            "repeats": repeats,
        },
        "environment": environment,
        "physics": {
            "successful": successful,
            "all_points_in_declared_domain": bool(jnp.all(primal[4])),
            "all_finite": bool(jnp.all(primal[5])),
            "all_physically_valid": bool(jnp.all(primal[6])),
            "all_derivatives_valid": bool(jnp.all(primal[7])),
            "curvature_invariants_finite": curvature_invariants_finite,
            "domain_status_counts": {
                str(int(status)): int(count)
                for status, count in zip(status_values, status_counts, strict=True)
            },
            "maximum_metric_symmetry_residual": float(symmetry_residual),
            "maximum_determinant_relative_error": float(
                determinant_relative_error
            ),
            "kretschmann_range": [
                float(jnp.min(primal[2])),
                float(jnp.max(primal[2])),
            ],
            "pontryagin_range": [
                float(jnp.min(primal[3])),
                float(jnp.max(primal[3])),
            ],
            "directional_derivatives_finite": tangent_finite,
        },
        "performance": {
            "setup_seconds": setup_seconds,
            "primal": primal_performance,
            "directional_derivative": derivative_performance,
            "logical_bytes": {
                "metric": logical_array_bytes(metric),
                "input_points": logical_array_bytes(points),
                "input_direction": logical_array_bytes(direction),
                "primal_output": logical_array_bytes(primal),
                "derivative_output": logical_array_bytes(tangents),
            },
            "points_per_second": None
            if mean_seconds in (None, 0.0)
            else point_count / mean_seconds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not 1 <= arguments.points <= 1_000_000:
        raise ValueError("points must be between 1 and 1,000,000.")
    if not 0 <= arguments.warmup <= 100:
        raise ValueError("warmup must be between 0 and 100.")
    if not 1 <= arguments.repeats <= 1_000:
        raise ValueError("repeats must be between 1 and 1,000.")
    payload = run(arguments.points, arguments.warmup, arguments.repeats)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["physics"]["successful"] else 1)


if __name__ == "__main__":
    main()
