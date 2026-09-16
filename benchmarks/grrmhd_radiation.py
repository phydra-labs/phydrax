#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark conservative implicit GRRMHD radiation-matter source coupling."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)

import phydrax as phx
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


def _setup(cell_count: int):
    scale = phx.RelativityScaleContract.geometric(KILOGRAM)
    convention = phx.metrix.RelativityConvention.canonical()
    eos = phx.equations.GammaLawEOS(scale, 4.0 / 3.0, minimum_density=1.0e-12)
    material = phx.equations.IdealValenciaGRMHDSystem(
        eos,
        scale,
        convention=convention,
        maximum_magnetization=1.0e6,
        recovery_iterations=32,
        enthalpy_iterations=32,
    )
    radiation = phx.equations.GRGreyM1RadiationSystem(scale, convention)
    opacity = phx.equations.ConstantGRGreyOpacityPlan(
        planck_absorption=0.5,
        planck_emission=0.0,
        rosseland_transport=0.5,
        scattering=0.1,
    )
    interaction = phx.equations.GRGreyRadiationInteractionPlan(radiation, opacity)
    source = phx.solver.GRRMHDImplicitSourcePlan(
        material,
        interaction,
        maximum_iterations=24,
        caloric_temperature_scale=1.0,
    )
    shape = (cell_count,)
    identity = jnp.broadcast_to(jnp.eye(3), shape + (3, 3))
    geometry = phx.metrix.ADMGridGeometry(
        jnp.ones(shape),
        jnp.zeros(shape + (3,)),
        identity,
        identity,
        jnp.ones(shape),
        jnp.zeros(shape + (3, 3)),
        jnp.ones(shape, dtype=bool),
        jnp.ones(shape, dtype=bool),
        snapshot_token=jnp.asarray(0, dtype=jnp.int32),
        chart_id="benchmark-cartesian",
        convention_id=convention.convention_id,
        scale_id=scale.scale_id,
        topology_id="grrmhd-source-line",
        geometry_lineage_id="minkowski-line",
    )
    phase = 2.0 * jnp.pi * (jnp.arange(cell_count) + 0.5) / cell_count
    primitive = jnp.zeros(shape + (8,))
    primitive = primitive.at[..., 0].set(1.0 + 0.05 * jnp.sin(phase))
    primitive = primitive.at[..., 1].set(0.02 * jnp.cos(phase))
    primitive = primitive.at[..., 4].set(0.2)
    primitive = primitive.at[..., 5].set(0.1)
    material_state = material.primitive_to_conserved(primitive, geometry)
    moments = jnp.stack(
        (
            2.0 + 0.1 * jnp.cos(phase),
            0.05 * jnp.sin(phase),
            jnp.zeros_like(phase),
            jnp.zeros_like(phase),
        ),
        axis=-1,
    )
    radiation_state = geometry.sqrt_det_spatial_metric[..., None] * moments
    return material, radiation, source, geometry, material_state, radiation_state


def run(cell_count: int, warmup: int, repeats: int) -> dict[str, object]:
    environment = capture_environment().to_dict()
    setup, setup_seconds = measure_synchronized(lambda: _setup(cell_count))
    material, radiation, source, geometry, material_state, radiation_state = setup

    def source_kernel(material_values, radiation_values):
        result = source.advance(
            material_values,
            radiation_values,
            jnp.asarray(0.05, dtype=material_values.dtype),
            geometry,
        )
        final_moments = (
            result.radiation_state / geometry.sqrt_det_spatial_metric[..., None]
        )
        closure = radiation.closure(
            final_moments[..., 0], final_moments[..., 1:], geometry
        )
        return (
            result.material_state,
            result.radiation_state,
            result.ledger.energy_defect,
            result.ledger.momentum_defect,
            result.ledger.maximum_residual,
            result.accepted,
            result.finite,
            result.converged,
            result.physically_valid,
            result.qualified,
            result.derivative_valid,
            closure.qualified,
        )

    result, performance = _measure(
        source_kernel,
        (material_state, radiation_state),
        warmup,
        repeats,
    )
    energy_defect = jnp.max(jnp.abs(result[2]), initial=0.0)
    momentum_defect = jnp.max(jnp.abs(result[3]), initial=0.0)
    tolerance = 512.0 * jnp.finfo(material_state.dtype).eps
    successful = bool(
        result[5]
        & result[6]
        & result[7]
        & result[8]
        & result[9]
        & result[10]
        & jnp.all(result[11])
        & (energy_defect <= tolerance)
        & (momentum_defect <= tolerance)
    )
    mean_seconds = performance["execution"]["mean_seconds"]
    return {
        "identities": {
            "benchmark": "grrmhd-radiation",
            "kernel": "implicit-conservative-four-force-source",
            "material_system": material.system_id,
            "radiation_system": radiation.system_id,
            "source_plan": source.plan_id,
        },
        "configuration": {
            "cell_capacity": cell_count,
            "source_step": 0.05,
            "warmup": warmup,
            "repeats": repeats,
        },
        "environment": environment,
        "physics": {
            "successful": successful,
            "accepted": bool(result[5]),
            "finite": bool(result[6]),
            "converged": bool(result[7]),
            "physically_valid": bool(result[8]),
            "qualified": bool(result[9]),
            "derivative_valid": bool(result[10]),
            "radiation_realizable": bool(jnp.all(result[11])),
            "maximum_energy_balance_defect": float(energy_defect),
            "maximum_momentum_balance_defect": float(momentum_defect),
            "maximum_nonlinear_residual": float(result[4]),
            "balance_tolerance": float(tolerance),
        },
        "performance": {
            "setup_seconds": setup_seconds,
            "source": performance,
            "logical_bytes": {
                "material_input": logical_array_bytes(material_state),
                "radiation_input": logical_array_bytes(radiation_state),
                "source_output": logical_array_bytes(result),
            },
            "cell_sources_per_second": None
            if mean_seconds in (None, 0.0)
            else cell_count / mean_seconds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=1_024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not 1 <= arguments.cells <= 1_000_000:
        raise ValueError("cells must be between 1 and 1,000,000.")
    if not 0 <= arguments.warmup <= 100:
        raise ValueError("warmup must be between 0 and 100.")
    if not 1 <= arguments.repeats <= 1_000:
        raise ValueError("repeats must be between 1 and 1,000.")
    payload = run(arguments.cells, arguments.warmup, arguments.repeats)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["physics"]["successful"] else 1)


if __name__ == "__main__":
    main()
