#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark bounded Kerr ray tracing followed by scalar radiative transfer."""

from __future__ import annotations

import argparse
import json
import math
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
from phydrax.applications import astrophysics
from phydrax.units import KILOGRAM


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


def _pixel_coordinates(pixel_count: int) -> np.ndarray:
    side = math.ceil(math.sqrt(pixel_count))
    axis = np.linspace(-0.12, 0.12, side)
    horizontal, vertical = np.meshgrid(axis, axis, indexing="xy")
    return np.stack((horizontal.ravel(), vertical.ravel()), axis=-1)[:pixel_count]


def _setup(pixel_count: int, sample_count: int, maximum_steps: int):
    chart = phx.metrix.CoordinateChart(
        "benchmark-kerr-boyer-lindquist", ("t", "r", "theta", "phi")
    )
    scale = phx.RelativityScaleContract.geometric(KILOGRAM)
    convention = phx.metrix.RelativityConvention.canonical()
    coordinate_unit = scale.dimensional_scale.length_unit
    affine_parameter_unit = coordinate_unit
    mass = jnp.asarray(1.0)
    spin = jnp.asarray(0.5)
    metric = phx.metrix.kerr_boyer_lindquist_metric(
        mass, spin, chart=chart
    )
    screen_plan = astrophysics.GRObserverScreenPlan(
        metric,
        jnp.asarray((0.0, 12.0, 0.5 * jnp.pi, 0.0)),
        jnp.asarray((1.0, 0.0, 0.0, 0.0)),
        jnp.asarray((0.0, -1.0, 0.0, 0.0)),
        jnp.asarray((0.0, 0.0, -1.0, 0.0)),
        _pixel_coordinates(pixel_count),
        scale=scale,
        convention=convention,
        coordinate_unit=coordinate_unit,
        affine_parameter_unit=affine_parameter_unit,
        temporal_direction="past",
    )
    screen = screen_plan.initialize()
    affine = jnp.linspace(0.0, 6.0, sample_count)
    ray_plan = astrophysics.GRRayPlan.from_screen(
        metric,
        screen,
        affine,
        scale=scale,
        convention=convention,
        coordinate_unit=coordinate_unit,
        affine_parameter_unit=affine_parameter_unit,
        affine_budget_is_event=False,
        maximum_steps=maximum_steps,
    )
    segment_lengths = jnp.diff(affine, prepend=affine[:1])
    si_transfer_units = (
        astrophysics.InvariantTransferUnitContract.si_affine_length()
    )
    transfer_units = astrophysics.InvariantTransferUnitContract(
        affine_parameter_unit,
        si_transfer_units.invariant_stokes_unit,
    )
    transfer_plan = astrophysics.InvariantScalarTransferPlan(
        segment_lengths, transfer_units, path_id=ray_plan.plan_id
    )
    return (
        chart,
        metric,
        screen_plan,
        screen,
        ray_plan,
        transfer_units,
        transfer_plan,
        mass,
        spin,
    )


def run(
    pixel_count: int,
    sample_count: int,
    maximum_steps: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    environment = capture_environment().to_dict()
    setup, setup_seconds = measure_synchronized(
        lambda: _setup(pixel_count, sample_count, maximum_steps)
    )
    (
        chart,
        metric,
        screen_plan,
        screen,
        ray_plan,
        transfer_units,
        transfer_plan,
        mass,
        spin,
    ) = setup

    def imaging_kernel(emission_scale):
        rays = astrophysics.trace_gr_rays(ray_plan)
        radius = rays.coordinates[..., 1]
        support = rays.valid
        active = support.astype(radius.dtype)
        profile = active * jnp.exp(-0.5 * ((radius - 6.0) / 1.5) ** 2)
        extinction = active * jnp.asarray(0.03, dtype=radius.dtype)

        def transfer_one(emission, opacity, supported):
            result = transfer_plan.evaluate(
                emission, opacity, support=supported
            )
            return (
                result.invariant_intensity,
                result.optical_depth,
                result.evidence.finite,
                result.evidence.converged,
                result.evidence.physically_valid,
                result.evidence.in_support,
                result.evidence.qualified,
                result.evidence.derivative_valid,
            )

        transfer = jax.vmap(transfer_one)(
            emission_scale * profile, extinction, support
        )

        def image_total(scale):
            return jnp.sum(
                jax.vmap(
                    lambda emission, opacity, supported: transfer_plan.evaluate(
                        scale * emission, opacity, support=supported
                    ).invariant_intensity
                )(profile, extinction, support)
            )

        image_gradient = jax.grad(image_total)(emission_scale)
        return (
            rays.coordinates,
            rays.valid,
            rays.status,
            rays.event_code,
            rays.finite,
            rays.converged,
            rays.physically_valid,
            rays.qualified,
            rays.derivative_valid,
            rays.null_residual,
            transfer[0],
            transfer[1],
            transfer[2],
            transfer[3],
            transfer[4],
            transfer[5],
            transfer[6],
            transfer[7],
            image_gradient,
        )

    emission_scale = jnp.asarray(1.0)
    result, performance = _measure(
        imaging_kernel, (emission_scale,), warmup, repeats
    )
    status_values, status_counts = np.unique(
        np.asarray(result[2]), return_counts=True
    )
    event_values, event_counts = np.unique(
        np.asarray(result[3]), return_counts=True
    )
    valid_history = result[1]
    null_residual = jnp.max(
        jnp.where(valid_history, jnp.abs(result[9]), jnp.asarray(0.0))
    )
    mean_seconds = performance["execution"]["mean_seconds"]
    successful = bool(
        jnp.all(screen.valid)
        & jnp.all(result[4])
        & jnp.all(result[5])
        & jnp.all(result[6])
        & jnp.all(result[7])
        & jnp.all(result[8])
        & jnp.all(result[12])
        & jnp.all(result[13])
        & jnp.all(result[14])
        & jnp.all(result[15])
        & jnp.all(result[16])
        & jnp.all(result[17])
        & jnp.all(jnp.isfinite(result[10]))
        & jnp.isfinite(result[18])
    )
    return {
        "identities": {
            "benchmark": "gr-imaging",
            "kernel": "kerr-null-ray-trace-plus-scalar-transfer",
            "chart": chart.name,
            "metric_convention": metric.convention,
            "metric": ray_plan.metric_id,
            "chart_context": ray_plan.chart_id,
            "convention": ray_plan.convention_id,
            "scale": ray_plan.scale_id,
            "coordinate_unit": ray_plan.coordinate_unit_id,
            "affine_parameter_unit": ray_plan.affine_parameter_unit_id,
            "screen_plan": screen_plan.plan_id,
            "screen_result": screen.result_id,
            "ray_plan": ray_plan.plan_id,
            "transfer_plan": transfer_plan.plan_id,
            "transfer_units": transfer_units.units_id,
        },
        "configuration": {
            "pixel_capacity": pixel_count,
            "history_sample_capacity": sample_count,
            "maximum_solver_steps_per_ray": maximum_steps,
            "mass_in_path_parameter_units": float(mass),
            "spin_in_path_parameter_units": float(spin),
            "affine_interval": [
                float(ray_plan.affine_parameter[0]),
                float(ray_plan.affine_parameter[-1]),
            ],
            "transfer_parameterization": "prescribed affine-parameter segments",
            "emission_profile": "nonnegative synthetic radial Gaussian",
            "path_parameter_unit": transfer_units.path_parameter_unit.symbol,
            "coordinate_unit": ray_plan.coordinate_unit.symbol,
            "affine_parameter_unit": ray_plan.affine_parameter_unit.symbol,
            "terminal_events": "disabled for the fixed exterior path profile",
            "warmup": warmup,
            "repeats": repeats,
        },
        "environment": environment,
        "physics": {
            "successful": successful,
            "screen_all_valid": bool(jnp.all(screen.valid)),
            "screen_tetrad_residual": float(screen.tetrad_residual),
            "screen_maximum_null_residual": float(jnp.max(screen.null_residual)),
            "rays_all_finite": bool(jnp.all(result[4])),
            "rays_all_converged": bool(jnp.all(result[5])),
            "rays_all_physically_valid": bool(jnp.all(result[6])),
            "rays_all_qualified": bool(jnp.all(result[7])),
            "rays_all_derivatives_valid": bool(jnp.all(result[8])),
            "maximum_valid_history_null_residual": float(null_residual),
            "ray_status_counts": {
                str(int(status)): int(count)
                for status, count in zip(status_values, status_counts, strict=True)
            },
            "ray_event_counts": {
                str(int(event)): int(count)
                for event, count in zip(event_values, event_counts, strict=True)
            },
            "transfer_all_finite": bool(jnp.all(result[12])),
            "transfer_all_converged": bool(jnp.all(result[13])),
            "transfer_all_physically_valid": bool(jnp.all(result[14])),
            "transfer_all_in_support": bool(jnp.all(result[15])),
            "transfer_all_qualified": bool(jnp.all(result[16])),
            "transfer_all_derivatives_valid": bool(jnp.all(result[17])),
            "image_intensity_range": [
                float(jnp.min(result[10])),
                float(jnp.max(result[10])),
            ],
            "optical_depth_range": [
                float(jnp.min(result[11])),
                float(jnp.max(result[11])),
            ],
            "emission_scale_image_derivative": float(result[18]),
            "emission_scale_derivative_finite": bool(jnp.isfinite(result[18])),
        },
        "performance": {
            "setup_seconds": setup_seconds,
            "imaging": performance,
            "logical_bytes": {
                "screen_plan": logical_array_bytes(screen_plan),
                "screen": logical_array_bytes(screen),
                "ray_plan": logical_array_bytes(ray_plan),
                "transfer_plan": logical_array_bytes(transfer_plan),
                "emission_scale": logical_array_bytes(emission_scale),
                "output": logical_array_bytes(result),
            },
            "ray_history_samples_per_second": None
            if mean_seconds in (None, 0.0)
            else pixel_count * sample_count / mean_seconds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=16)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--maximum-steps", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not 1 <= arguments.pixels <= 4_096:
        raise ValueError("pixels must be between 1 and 4,096.")
    if not 2 <= arguments.samples <= 1_024:
        raise ValueError("samples must be between 2 and 1,024.")
    if not 1 <= arguments.maximum_steps <= 65_536:
        raise ValueError("maximum-steps must be between 1 and 65,536.")
    if not 0 <= arguments.warmup <= 100:
        raise ValueError("warmup must be between 0 and 100.")
    if not 1 <= arguments.repeats <= 1_000:
        raise ValueError("repeats must be between 1 and 1,000.")
    payload = run(
        arguments.pixels,
        arguments.samples,
        arguments.maximum_steps,
        arguments.warmup,
        arguments.repeats,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["physics"]["successful"] else 1)


if __name__ == "__main__":
    main()
