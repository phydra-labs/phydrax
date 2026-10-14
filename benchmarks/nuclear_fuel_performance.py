#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Measured assembled fuel runtime scaling; no experimental validation claim."""

from __future__ import annotations

import argparse
import json
import time as clock
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._runtime import (
    benchmark_driver_fingerprint,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
    RuntimeEnvironment,
    source_build_fingerprint,
)
from phydrax import config as pxconfig
from phydrax._fingerprint import canonical_fingerprint
from phydrax.applications.nuclear_fuel import (
    fuel_parameterization,
    FuelFixedGrid,
    FuelGeometryReduction,
    FuelNumericalExecution,
    FuelParameterName,
    FuelRodSimulation,
    FuelRodState,
    FuelSensitivityPlan,
    FuelStepEvidence,
    nuclear_fuel_execution_profile,
)
from phydrax.qualification._evidence import ObservedResourceRecord
from tools.nuclear_fuel_qualification import (
    bitwise_equal_tree,
    fuel_step_failure_record,
    prepare_numerical_rod,
    whole_history_balances,
)


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))


@dataclass(frozen=True, slots=True)
class FuelBenchmarkCase:
    scaling_axis: str
    reduction: FuelGeometryReduction = "layered-radial"
    radial_cells: int = 4
    axial_layers: int = 1
    axial_cells_per_zone: int = 1
    modes: int = 8
    helium_radial_cells: int = 40
    parameters: int = 1
    base_duration: float = 3.0e7
    base_linear_power: float = 20000.0
    transient_duration: float = 3600.0


def benchmark_cases(smoke: bool, /) -> tuple[FuelBenchmarkCase, ...]:
    """Vary each capacity independently; never imply a timed unexecuted tuple."""
    base = FuelBenchmarkCase("baseline")
    if smoke:
        base = replace(
            base,
            radial_cells=2,
            modes=4,
            base_duration=3480.0,
            base_linear_power=3000.0,
            transient_duration=60.0,
        )
    radial_counts = (4, 8) if smoke else (8, 16)
    layer_counts = (2, 3) if smoke else (3, 6)
    mode_counts = (8, 16) if smoke else (16, 32)
    return (
        base,
        *(
            replace(base, scaling_axis="radial-cells", radial_cells=n)
            for n in radial_counts
        ),
        *(
            replace(base, scaling_axis="axial-layers", axial_layers=n)
            for n in layer_counts
        ),
        *(
            replace(
                base,
                scaling_axis="rz-capacity",
                reduction="axisymmetric-rz",
                axial_layers=1 if smoke else 3,
                axial_cells_per_zone=n,
            )
            for n in (1, 2, 4)
        ),
        *(replace(base, scaling_axis="fission-mode-count", modes=n) for n in mode_counts),
        *(replace(base, scaling_axis="parameter-count", parameters=n) for n in (2, 4, 8)),
        *(
            replace(base, scaling_axis="helium-radial-cells", helium_radial_cells=n)
            for n in (80, 160)
        ),
    )


def benchmark_case(
    case: FuelBenchmarkCase,
    directory: Path,
    /,
    *,
    warmup: int,
    repeats: int,
    environment: RuntimeEnvironment,
    build_id: str,
) -> dict[str, object]:
    prepared, preparation = measure_synchronized(
        lambda: prepare_numerical_rod(
            fuel_cells=case.radial_cells,
            axial_layers=case.axial_layers,
            axial_cells_per_zone=case.axial_cells_per_zone,
            mode_count=case.modes,
            helium_radial_cell_count=case.helium_radial_cells,
            reduction=case.reduction,
            base_duration=case.base_duration,
            base_linear_power=case.base_linear_power,
            transient_duration=case.transient_duration,
        )
    )
    initial, initialization = measure_synchronized(prepared.initial_state)
    # The production numerical owner carries every changing array explicitly;
    # scientific receipt metadata never enters the compiled argument tree.
    numerical, numerical_preparation = measure_host(
        lambda: FuelNumericalExecution(prepared)
    )
    binding, binding_preparation = measure_host(lambda: numerical.bind(prepared))
    step_size = min(60.0, case.transient_duration / max(100, warmup + repeats + 2))
    dt = jnp.asarray(step_size, dtype=initial.time.dtype)
    arguments = (
        binding.numerical_inputs,
        binding.bundle_inputs,
        tuple(jax.tree.leaves(initial)),
        jnp.asarray(0, dtype=jnp.int32),
        initial.time,
        dt,
    )
    executable, compilation = measure_lower_and_compile(
        lambda: numerical.compiled_step.lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    first_arrays, first_seconds = measure_synchronized(lambda: executable(*arguments))
    first = binding.restore_result(first_arrays)
    first_evidence = first.evidence
    if not isinstance(first_evidence, FuelStepEvidence):
        raise TypeError(
            "Benchmark numerical step must retain native fuel acceptance evidence."
        )
    if not isinstance(first.accepted_state, FuelRodState):
        raise TypeError(
            "Benchmark numerical step must return a complete accepted rod state."
        )
    if not bool(first.successful):
        raise RuntimeError(
            f"Benchmark native step failed: {fuel_step_failure_record(first_evidence)}"
        )
    timed_states: list[FuelRodState] = [first.accepted_state]
    state = first.accepted_state
    step_index = 1

    def execute_next_step() -> tuple[jax.Array, ...]:
        nonlocal state, step_index
        arrays = executable(
            binding.numerical_inputs,
            binding.bundle_inputs,
            tuple(jax.tree.leaves(state)),
            jnp.asarray(step_index, dtype=jnp.int32),
            state.time,
            dt,
        )
        result = binding.restore_result(arrays)
        if not isinstance(result.accepted_state, FuelRodState):
            raise TypeError(
                "Benchmark numerical step must return a complete accepted rod state."
            )
        evidence = result.evidence
        if not isinstance(evidence, FuelStepEvidence):
            raise TypeError(
                "Benchmark changing-state step omitted native fuel acceptance evidence."
            )
        if not bool(result.successful):
            raise RuntimeError(
                f"Benchmark changing-state native step failed physical acceptance: {fuel_step_failure_record(evidence)}"
            )
        state = result.accepted_state
        timed_states.append(state)
        step_index += 1
        return arrays

    _, warm = measure_repeated(execute_next_step, warmup=warmup, repeats=repeats)
    if numerical.trace_count != 1:
        raise RuntimeError(
            "Benchmark changing-state entry point did not retain one numerical trace."
        )
    compiler = compiler_evidence(
        executable.cost_analysis(),
        executable.memory_analysis(),
        source="xla",
        unavailable_reason="Selected backend did not supply all requested compiler estimates.",
    )
    simulation, lifecycle_preparation = measure_host(
        lambda: FuelRodSimulation(
            prepared,
            time_step=120.0,
            checkpoint_directory=directory / "history",
            checkpoint_interval=1,
            segment_steps=1,
            minimum_time_step=0.01,
            maximum_time_step=max(480.0, case.base_duration / 100),
            numerical_execution=numerical,
        )
    )
    result, history_seconds = measure_synchronized(simulation.run)
    if not bool(result.runtime_result.successful):
        diagnostics = tuple(
            fuel_step_failure_record(attempt)
            for attempt in result.attempts
            if not bool(attempt.accepted)
        )
        raise RuntimeError(
            f"Benchmark history failed: {result.runtime_result.failure}; attempts={diagnostics}"
        )
    balances, check_seconds = measure_host(
        lambda: whole_history_balances(result, prepared)
    )
    if (
        balances["maximum_energy_relative_error"] > 1e-6
        or balances["maximum_gas_relative_error"] > 1e-6
    ):
        raise RuntimeError("Benchmark physical balance check failed.")
    # Checkpoint commit and restart loading are measured independently of step
    # execution and of the full-history timing, using an actual accepted state.
    checkpointed, checkpoint_seconds = measure_synchronized(
        lambda: simulation.runtime.checkpoint(result.runtime_result.state)
    )
    resumed, restart_seconds = measure_synchronized(
        lambda: simulation.runtime.resume(checkpointed)
    )
    if not bitwise_equal_tree(checkpointed.accepted_state, resumed.accepted_state):
        raise RuntimeError("Benchmark restart changed the accepted physical state.")
    names: tuple[FuelParameterName, ...] = (
        "fuel-phonon-resistivity",
        "fuel-phonon-temperature-coefficient",
        "fuel-electronic-coefficient",
        "fuel-electronic-temperature",
        "fuel-porosity-normalization",
        "fuel-pore-shape-coefficient",
        "gas-intrinsic-diffusivity-prefactor",
        "gas-intrinsic-activation-energy",
    )
    parameterization, values = fuel_parameterization(
        prepared.admission.bundle, names[: case.parameters]
    )
    grid = FuelFixedGrid(
        tuple(float(state.time) for state in result.states), prepared.prepared_id
    )
    tangent_plan, tangent_preparation = measure_host(
        lambda: FuelSensitivityPlan(
            prepared,
            grid,
            parameterization,
            values,
            parameter_capacity=8,
            maximum_working_set_bytes=268435456,
        )
    )
    tangents, tangent_seconds = measure_synchronized(
        lambda: tangent_plan.evaluate(result.states)
    )
    if not bool(tangents.successful):
        raise RuntimeError(
            "Benchmark fixed-grid tangent linear solve failed: "
            f"linear_success={np.asarray(tangents.linear_success).tolist()}, "
            f"linear_status={np.asarray(tangents.linear_status).tolist()}, "
            f"branch_changed={np.asarray(tangents.branch_changed).tolist()}"
        )
    retained_prepared = logical_array_bytes(prepared)
    retained_history = logical_array_bytes(result)
    retained_tangents = logical_array_bytes(tangents)
    retained_numerical_inputs = logical_array_bytes(
        (binding.numerical_inputs, binding.bundle_inputs)
    )
    observed = {
        "preparation_seconds": preparation,
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "history_seconds": history_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "restart_seconds": restart_seconds,
        "tangent_seconds": tangent_seconds,
        "retained_prepared_bytes": retained_prepared,
        "retained_history_bytes": retained_history,
        "retained_tangent_bytes": retained_tangents,
        "retained_numerical_input_bytes": retained_numerical_inputs,
    }
    for name, measurement in (
        ("compiler_argument_bytes", compiler.argument_bytes),
        ("compiler_temporary_bytes", compiler.temporary_bytes),
        ("compiler_output_bytes", compiler.output_bytes),
        ("compiler_generated_code_bytes", compiler.generated_code_bytes),
    ):
        if measurement is not None:
            observed[name] = measurement
    resource_artifact_id = canonical_fingerprint(observed)
    resource = ObservedResourceRecord(
        prepared.prepared_id,
        build_id,
        environment.fingerprint,
        backend=environment.backend,
        topology=prepared.geometry.geometry_id,
        measurements=observed,
        observed_at=int(clock.time()),
        raw_artifact_ids=(resource_artifact_id,),
    )
    precision_dtype = str(initial.fields.temperature_fuel.dtype)
    precision = {
        "dtype": precision_dtype,
        "native_nonlinear_residual": float(first_evidence.solver.nonlinear.residual),
        "native_coupling_joint_residual": float(
            first_evidence.solver.coupling.joint_residual
        ),
        "energy_relative_error": balances["maximum_energy_relative_error"],
        "gas_relative_error": balances["maximum_gas_relative_error"],
    }
    precision_id = canonical_fingerprint(precision)
    profile = nuclear_fuel_execution_profile(
        prepared.admission.bundle,
        reduction=case.reduction,
        coolant="prescribed-wall-temperature",
        axial_interaction="frictionless",
        prepared_id=prepared.prepared_id,
        runtime_id=simulation.runtime.run_id,
        resource_evidence_id=resource.record_id,
        precision_evidence_id=precision_id,
        dtype=precision_dtype,
    )
    simulation.close()
    return {
        "scaling_axis": case.scaling_axis,
        "reduction": case.reduction,
        "radial_cells": case.radial_cells,
        "axial_layers": case.axial_layers,
        "rz_axial_cells_per_zone": case.axial_cells_per_zone,
        "base_duration_seconds": case.base_duration,
        "base_linear_power_W_per_m": case.base_linear_power,
        "transient_duration_seconds": case.transient_duration,
        "fission_gas_modes": case.modes,
        "helium_radial_cells": case.helium_radial_cells,
        "parameter_count": case.parameters,
        "site_count": prepared.geometry.fuel.site_count
        + prepared.geometry.cladding.site_count,
        "prepared_id": prepared.prepared_id,
        "bundle_id": prepared.admission.bundle.bundle_id,
        "runtime_run_id": result.runtime_result.run_id,
        "preparation_seconds": preparation,
        "initialization_seconds": initialization,
        "numerical_execution_preparation_seconds": numerical_preparation,
        "numerical_binding_preparation_seconds": binding_preparation,
        "lifecycle_preparation_seconds": lifecycle_preparation,
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "first_step_seconds": first_seconds,
        "warm_step": warm.to_dict(),
        "timed_step_size_seconds": step_size,
        "timed_step_entry": "compiled-native-step; warm samples include host result binding/acceptance",
        "timed_step_state_times": [float(state.time) for state in timed_states],
        "numerical_execution_id": numerical.execution_id,
        "numerical_trace_count": numerical.trace_count,
        "history_seconds": history_seconds,
        "balance_check_seconds": check_seconds,
        "history_entry": "case-bound-production-runtime; includes segment lowering/compilation",
        "checkpoint_write_seconds": checkpoint_seconds,
        "restart_load_seconds": restart_seconds,
        "tangent_preparation_seconds": tangent_preparation,
        "tangent_seconds": tangent_seconds,
        "compiler": {
            "argument_bytes": compiler.argument_bytes,
            "temporary_bytes": compiler.temporary_bytes,
            "output_bytes": compiler.output_bytes,
            "generated_code_bytes": compiler.generated_code_bytes,
            "flops": compiler.flops,
            "bytes_accessed": compiler.bytes_accessed,
            "source": compiler.source,
            "unavailable_reason": compiler.unavailable_reason,
        },
        "retained_prepared_bytes": retained_prepared,
        "retained_numerical_input_bytes": retained_numerical_inputs,
        "retained_history_bytes": retained_history,
        "retained_tangent_bytes": retained_tangents,
        "tangent_working_set_bound_bytes": tangent_plan.maximum_working_set_bytes,
        "tangent_working_set_forecast_bytes": tangent_plan.working_set_bytes,
        "attempted_steps": len(result.attempts),
        "accepted_steps": len(result.states) - 1,
        "thermomechanical_iterations": sum(
            int(attempt.solver.coupling.thermomechanical_iterations)
            for attempt in result.attempts
        ),
        "thermomechanical_residual_evaluations": sum(
            int(attempt.solver.coupling.thermomechanical_residual_evaluations)
            for attempt in result.attempts
        ),
        "coupling_evaluations": sum(
            int(attempt.solver.coupling.evaluations) for attempt in result.attempts
        ),
        "gas_block_evaluations": sum(
            int(attempt.solver.coupling.gas_block_evaluations)
            for attempt in result.attempts
        ),
        "nonlinear_work_counts_complete": all(
            bool(attempt.solver.nonlinear.counts_complete) for attempt in result.attempts
        ),
        "branch_change_steps": np.flatnonzero(
            np.asarray(tangents.branch_changed)
        ).tolist(),
        "balance_evidence": balances,
        "native_terminal_status": result.runtime_result.state.status,
        "scope": "synthetic-numerical-equation-reproduction-not-validation",
        "exact_execution_profile": profile.to_record(),
        "observed_resource_record": resource.to_record(),
        "resource_raw_artifact": {
            "raw_artifact_id": resource_artifact_id,
            "measurements": observed,
        },
        "precision_raw_artifact": {
            "raw_artifact_id": precision_id,
            "measurements": precision,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--case-index",
        type=int,
        nargs="+",
        help="Run selected zero-based capacity cases in canonical matrix order.",
    )
    arguments = parser.parse_args()
    cases = benchmark_cases(arguments.smoke)
    selected = (
        tuple(range(len(cases)))
        if arguments.case_index is None
        else tuple(arguments.case_index)
    )
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(index < 0 or index >= len(cases) for index in selected)
    ):
        parser.error("--case-index requires unique existing zero-based matrix indices.")
    environment = capture_environment()
    root = Path(__file__).resolve().parents[1]
    build_id = source_build_fingerprint(root)
    drivers = (Path(__file__), root / "tools" / "nuclear_fuel_qualification.py")
    driver_id = canonical_fingerprint(
        tuple(benchmark_driver_fingerprint(root, path) for path in drivers)
    )
    with TemporaryDirectory(prefix="phydrax-fuel-benchmark-") as temporary:
        rows = [
            benchmark_case(
                case,
                Path(temporary) / str(index),
                warmup=arguments.warmup,
                repeats=arguments.repeats,
                environment=environment,
                build_id=build_id,
            )
            for index, case in enumerate(cases)
            if index in selected
        ]
    if (
        source_build_fingerprint(root) != build_id
        or canonical_fingerprint(
            tuple(benchmark_driver_fingerprint(root, path) for path in drivers)
        )
        != driver_id
    ):
        raise RuntimeError(
            "Package/driver sources changed during the benchmark; measured build identity is not stable."
        )
    report = {
        "environment": environment.to_dict(),
        "build_id": build_id,
        "driver_id": driver_id,
        "rows": rows,
        "scientific_release_status": "candidate-unreleased",
        "external_prerequisites": (
            "IFPE-originator-data-agreement",
            "independent-locked-evaluation",
        ),
    }
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
