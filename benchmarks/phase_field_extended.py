#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
)


def _square_mesh():
    return phx.discretization.CellMesh.from_triangles(
        jnp.asarray(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            dtype=jnp.float64,
        ),
        jnp.asarray(((0, 1, 3), (1, 2, 3)), dtype=jnp.int32),
    )


def _compiler_report(executable) -> dict[str, object]:
    cost = executable.compiled.cost_analysis()
    memory = executable.compiled.memory_analysis()
    evidence = compiler_evidence(
        cost,
        memory,
        source="jax-lowered-compiled-executable",
        unavailable_reason=(
            "Compiler analysis unavailable." if not cost and memory is None else None
        ),
    )
    return {
        **asdict(evidence),
        "estimated_device_memory_bytes": evidence.estimated_device_memory_bytes,
    }


def _compiled_step_case(method, state, step_size, repeats: int, /):
    arguments = (
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float64),
        state,
        jnp.asarray(step_size, dtype=jnp.float64),
        None,
    )
    kernel: Any = eqx.filter_jit(method.step)
    executable, compilation = measure_lower_and_compile(
        lambda: kernel.lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    result, durations = measure_repeated(
        lambda: executable(*arguments),
        warmup=1,
        repeats=repeats,
    )
    return result, {
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "steady_step": durations.to_milliseconds_dict(),
        "compiler": _compiler_report(executable),
        "logical_argument_bytes": logical_array_bytes(arguments),
        "logical_result_bytes": logical_array_bytes(result),
    }


def _binary_case(repeats: int) -> dict[str, object]:
    element = phx.discretization.lagrange_element("triangle", 1)
    mesh = _square_mesh()
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        (
            phx.discretization.FiniteElementFieldSpec("c", element),
            phx.discretization.FiniteElementFieldSpec("mu", element),
        ),
    ).prepare()
    model = phx.applications.phase_field.BinaryPhaseFieldModel(
        phx.equations.BinaryThermodynamicParameters(1.0, 1.0),
        closure=phx.equations.BinaryPhaseThermodynamicClosure(
            phx.equations.PolynomialBulkFreeEnergy((0.25, 0.0, -0.5, 0.0, 0.25))
        ),
    )
    mobility = phx.applications.phase_field.TensorPhaseFieldMobility(
        jnp.asarray(((1.0, 0.0), (0.0, 0.5)), dtype=jnp.float64)
    )
    method, preparation_seconds = measure_host(
        lambda: phx.applications.phase_field.CahnHilliardFEMPlan(model, mobility).prepare(
            discretization, "c", "mu"
        )
    )
    state = method.initialize(jnp.asarray((-0.2, 0.1, 0.3, -0.1), dtype=jnp.float64))
    result, timing = _compiled_step_case(method, state, 0.005, repeats)
    return {
        "name": "anisotropic-general-potential-cahn-hilliard",
        "preparation_seconds": preparation_seconds,
        "successful": bool(result.successful),
        "iterations": int(np.asarray(result.iterations)),
        "residual": float(np.asarray(result.residual)),
        "energy_before": float(np.asarray(state.energy)),
        "energy_after": float(np.asarray(result.candidate_state.energy)),
        **timing,
    }


def _grand_case(repeats: int) -> dict[str, object]:
    phase_a = phx.applications.phase_field.QuadraticGrandPotentialPhase(
        "benchmark-a", 0.0, jnp.asarray((0.2,)), jnp.asarray(((1.0,),))
    )
    phase_b = phx.applications.phase_field.QuadraticGrandPotentialPhase(
        "benchmark-b", 0.0, jnp.asarray((0.8,)), jnp.asarray(((1.0,),))
    )
    catalog = phx.applications.phase_field.GrandPotentialMaterialCatalog(
        (phase_a, phase_b)
    )
    model = phx.applications.phase_field.GrandPotentialMixtureModel(
        catalog,
        barrier_scale=0.1,
        gradient_coefficient=0.2,
        kinetic_coefficient=1.0,
        mobility=1.0,
    )
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _square_mesh(),
        (
            phx.discretization.FiniteElementFieldSpec(
                "eta", element, component_shape=(2,)
            ),
            phx.discretization.FiniteElementFieldSpec(
                "mu", element, component_shape=(1,)
            ),
        ),
    ).prepare()
    method, preparation_seconds = measure_host(
        lambda: phx.applications.phase_field.GrandPotentialFEMPlan(
            model,
            absolute_energy_tolerance=1.0,
            relative_energy_tolerance=1.0,
            component_tolerance=1.0e-7,
        ).prepare(discretization, "eta", "mu")
    )
    state = method.initialize(
        jnp.asarray(
            ((1.0, -1.0), (0.5, -0.5), (-0.5, 0.5), (-1.0, 1.0)),
            dtype=jnp.float64,
        ),
        jnp.zeros((4, 1), dtype=jnp.float64),
    )
    result, timing = _compiled_step_case(method, state, 1.0e-3, repeats)
    return {
        "name": "dense-grand-potential",
        "preparation_seconds": preparation_seconds,
        "successful": bool(result.successful),
        "iterations": int(np.asarray(result.iterations)),
        "residual": float(np.asarray(result.residual)),
        "component_defect": float(
            np.asarray(
                jnp.max(
                    jnp.abs(
                        result.candidate_state.components - state.reference_components
                    )
                )
            )
        ),
        **timing,
    }


def _structural_cases(repeats: int) -> dict[str, Any]:
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _square_mesh(),
        phx.discretization.FiniteElementFieldSpec("eta", element),
    ).prepare()
    active = phx.applications.phase_field.ActivePhaseStoragePlan(
        discretization.dof_maps[0].cell_dofs[0],
        16,
        3,
        cell_phase_capacity=6,
    )
    dense = jnp.zeros((4, 16), dtype=jnp.float64)
    dense = dense.at[:, 0].set(0.6).at[:, 1].set(0.3).at[:, 2].set(0.1)
    active_kernel: Any = eqx.filter_jit(active.from_dense)
    active_result, active_timing = measure_repeated(
        lambda: active_kernel(dense),
        warmup=1,
        repeats=repeats,
    )
    model = phx.applications.phase_field.BinaryPhaseFieldModel(
        phx.equations.BinaryThermodynamicParameters(1.0, 1.0)
    )
    method = phx.applications.phase_field.AllenCahnFEMPlan(model, 1.0).prepare(
        discretization, "eta"
    )
    state = method.initialize(jnp.asarray((-0.2, 0.1, 0.3, -0.1), dtype=jnp.float64))
    epoch = phx.applications.phase_field.PhaseFieldAdaptiveEpoch(method, state)
    adaptation_plan = phx.applications.phase_field.PhaseFieldAdaptivityPlan(
        gradient_threshold=0.0,
        energy_tolerance=1.0,
    )
    adaptation, adaptation_seconds = measure_host(lambda: adaptation_plan.refine(epoch))
    distributed, distributed_seconds = measure_host(
        lambda: phx.applications.phase_field.DistributedPhaseFieldPlan(discretization, 2)
    )
    return {
        "active_storage": {
            "successful": bool(active_result.evidence.successful),
            "global_phase_count": active.phase_count,
            "local_capacity": active.local_capacity,
            "timing": active_timing.to_milliseconds_dict(),
        },
        "adaptation": {
            "successful": bool(adaptation.committed),
            "seconds": adaptation_seconds,
            "source_cells": discretization.mesh.blocks[0].cell_count,
            "target_cells": adaptation.candidate.method.discretization.mesh.blocks[
                0
            ].cell_count,
        },
        "distributed_preparation": {
            "valid": bool(
                distributed.part_count == 2
                and np.isfinite(
                    float(np.asarray(distributed.partition.evidence.imbalance_ratio))
                )
                and float(np.asarray(distributed.partition.evidence.imbalance_ratio))
                >= 1.0
            ),
            "seconds": distributed_seconds,
            "parts": distributed.part_count,
            "imbalance_ratio": float(
                np.asarray(distributed.partition.evidence.imbalance_ratio)
            ),
            "edge_cut": int(np.asarray(distributed.partition.evidence.edge_cut)),
        },
    }


def benchmark(*, quick: bool, repeats: int) -> dict[str, object]:
    if not bool(jax.config.read("jax_enable_x64")):
        raise ValueError("Extended phase-field benchmark requires float64.")
    repetitions = 1 if quick else repeats
    binary = _binary_case(repetitions)
    grand = _grand_case(repetitions)
    structural = _structural_cases(repetitions)
    passed = (
        binary["successful"]
        and grand["successful"]
        and structural["active_storage"]["successful"]
        and structural["adaptation"]["successful"]
        and structural["distributed_preparation"]["valid"]
    )
    return {
        "status": "pass" if passed else "fail",
        "environment": capture_environment().to_dict(),
        "settings": {"quick": quick, "repeats": repetitions},
        "compiled_cases": (binary, grand),
        "structural_cases": structural,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the integrated phase-field closure."
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/phase_field_extended.json"),
    )
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive.")
    report = benchmark(quick=arguments.quick, repeats=arguments.repeats)
    from benchmarks._io import write_json_atomic

    write_json_atomic(arguments.output, report)
    print(arguments.output)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
