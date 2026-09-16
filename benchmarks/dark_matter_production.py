#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import capture_environment
from phydrax.applications.cosmology._production_profiles import (
    PeriodicWaveProductionMethod,
)
from phydrax.applications.cosmology._simulation_products import (
    DarkMatterCheckpointContract,
    DarkMatterRestartSnapshot,
)
from phydrax.solver._production_runtime import (
    CheckpointGenerationPolicy,
    DurableCheckpointStore,
    PreparedProductionRun,
    ProductionCaseManifest,
)
from phydrax.solver._runtime_lifecycle import (
    ByteBoundedAsyncPublisher,
    ExactTimeSchedule,
    write_runtime_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--checkpoint-repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.size < 4
        or arguments.warmup < 0
        or arguments.repeats < 1
        or arguments.checkpoint_repeats < 1
    ):
        raise ValueError(
            "size >= 4, positive repeats, and nonnegative warmup are required"
        )

    cosmology = phx.applications.cosmology
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(arguments.size),),
        axis_names=("x",),
        field_name="psi",
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))
    background = cosmology.FLRWBackground(1.0, 1.0)
    prepared = cosmology.WaveDarkMatterPlan(
        1.0,
        jnp.asarray((0.5, 0.5001)),
        gravitational_constant=0.05,
        reduced_planck_constant=0.03,
        step_policy=cosmology.WaveDarkMatterStepPolicy(
            maximum_phase_radians=2.0,
            minimum_de_broglie_cells=2.0,
            norm_relative_tolerance=1.0e-6,
        ),
    ).prepare(space, background)
    x = space.axes[0].nodes
    psi = (1.0 + 0.01 * jnp.cos(2.0 * jnp.pi * x)).astype(jnp.complex128)
    initial_wave = prepared.initialize(psi)
    method = PeriodicWaveProductionMethod(prepared)
    output_schedule = ExactTimeSchedule(
        jnp.asarray((method.end_scale_factor,)),
        tolerance=method.schedule_tolerance,
    )
    plan = method.production_run_plan(
        checkpoint_interval=1,
        segment_steps=1,
        output_schedule=output_schedule,
    )
    support = space.support
    manifest = ProductionCaseManifest(
        problem_id=prepared.prepared_id,
        method_id=method.method_id,
        precision_id="complex128",
        topology_id=support.topology.topology_id,
        geometry_layout_id=support.embedding_id,
        dtype="complex128",
    )

    support_tuple = {
        "profile": "periodic-wave",
        "geometry": "fixed-tensor-fourier-grid",
        "boundary": "periodic",
        "backend": jax.default_backend(),
        "dtype": "complex128",
        "grid_points": arguments.size,
        "maximum_steps": 1,
        "partition": "serial",
        "fixed_capacity": True,
    }

    def execute_production(root: Path):
        published = []
        publisher = ByteBoundedAsyncPublisher(
            lambda event_id, snapshot: published.append(
                (event_id, float(np.asarray(snapshot.scale_factor)))
            ),
            maximum_pending=1,
            maximum_pending_bytes=16 * 1024 * 1024,
        )
        store = DurableCheckpointStore(
            root,
            manifest,
            CheckpointGenerationPolicy(2),
        )
        preparation_started = time.perf_counter()
        runtime = PreparedProductionRun(
            manifest,
            plan,
            store,
            publisher=publisher,
        )
        run_state = method.initial_run_state(runtime, initial_wave)
        preparation_seconds = time.perf_counter() - preparation_started
        execution_started = time.perf_counter()
        result = runtime.run(run_state)
        jax.block_until_ready(result.state.accepted_state.psi)
        execution_seconds = time.perf_counter() - execution_started
        publisher.close()
        checkpoint_bytes = sum(
            path.stat().st_size for path in root.glob("generation-*.phx")
        )
        store.close()
        return preparation_seconds, execution_seconds, result, published, checkpoint_bytes

    with tempfile.TemporaryDirectory(prefix="phydrax-dark-matter-production-") as root:
        directory = Path(root)
        first_preparation, first_total, first_result, first_outputs, first_bytes = (
            execute_production(directory / "compile")
        )
        for index in range(arguments.warmup):
            execute_production(directory / f"warmup-{index}")
        executions = []
        preparations = []
        results = []
        output_counts = []
        production_checkpoint_bytes = []
        for index in range(arguments.repeats):
            preparation, duration, result, outputs, checkpoint_bytes = execute_production(
                directory / f"run-{index}"
            )
            preparations.append(preparation)
            executions.append(duration)
            results.append(result)
            output_counts.append(len(outputs))
            production_checkpoint_bytes.append(checkpoint_bytes)

        result = results[-1]
        execution_mean = float(np.mean(executions))
        compilation_seconds = max(first_total - execution_mean, 0.0)
        restart = DarkMatterRestartSnapshot(
            result.state.accepted_state,
            time=result.state.time,
            accepted_step=result.state.step_index,
            schedule_cursor=result.state.schedule_cursor,
            output_cursor=result.state.output_cursor,
            accepted_evidence={"successful": result.successful},
            parent_checkpoint_id=result.state.last_checkpoint_id,
        )
        contract = DarkMatterCheckpointContract(
            profile_name="periodic-wave",
            physics_id=prepared.prepared_id,
            support_ids=(space.prepared_id,),
            source_ids=("native-periodic-wave",),
            interaction_ids=("self-gravity",),
            artifact_ids=("native-no-external-artifact",),
            topology_id=support.topology.topology_id,
            method_id=method.method_id,
            precision_id="complex128",
            restart_template=restart,
            topology_epoch_id=support.embedding_id,
            scale_id=prepared.scale_id,
            partition_id="serial",
        )
        checkpoint = contract.payload(restart)
        write_seconds = []
        read_seconds = []
        sizes = []
        restored = checkpoint
        for index in range(arguments.checkpoint_repeats):
            path = directory / f"portable-{index}.phx"
            started = time.perf_counter()
            # This non-root payload has durable ancestry from the completed production
            # run, so persist the exact native envelope for portable throughput timing.

            write_runtime_checkpoint(path, checkpoint.envelope)
            write_seconds.append(time.perf_counter() - started)
            sizes.append(path.stat().st_size)
            started = time.perf_counter()
            restored = contract.read(
                path,
                restart,
                expected_parent_checkpoint_id=result.state.last_checkpoint_id,
            )
            read_seconds.append(time.perf_counter() - started)

    checkpoint_bytes = int(max(sizes))
    mean_write = float(np.mean(write_seconds))
    mean_read = float(np.mean(read_seconds))
    production_success = all(bool(value.successful) for value in results)
    output_success = (
        all(count == 1 for count in output_counts) and len(first_outputs) == 1
    )
    checkpoint_success = (
        bool(restored.successful)
        and restored.envelope.checkpoint_id == checkpoint.envelope.checkpoint_id
    )
    successful = production_success and output_success and checkpoint_success
    poisson = prepared.poisson(result.state.accepted_state)
    payload = {
        "environment": capture_environment().to_dict(),
        "support_tuple": support_tuple,
        "identity": {
            "prepared": prepared.prepared_id,
            "method": method.method_id,
            "production_plan": plan.plan_id,
            "production_run": result.run_id,
            "checkpoint_contract": contract.contract_id,
            "checkpoint": checkpoint.envelope.checkpoint_id,
        },
        "compilation": {
            "runtime_preparation_seconds": first_preparation,
            "first_compile_and_execution_seconds": first_total,
            "estimated_compilation_seconds": compilation_seconds,
        },
        "execution": {
            "production_seconds_mean": execution_mean,
            "production_seconds_minimum": float(np.min(executions)),
            "production_seconds_maximum": float(np.max(executions)),
            "runtime_preparation_seconds_mean": float(np.mean(preparations)),
        },
        "capacity": {
            "grid_points": arguments.size,
            "active_grid_fraction": 1.0,
            "schedule_intervals": 1,
            "accepted_intervals": int(result.state.step_index),
            "output_events": output_counts[-1],
        },
        "resources": {
            "device_state_bytes": int(result.state.accepted_state.psi.size)
            * int(result.state.accepted_state.psi.dtype.itemsize),
            "communication_bytes": 0,
            "production_checkpoint_bytes_mean": float(
                np.mean(production_checkpoint_bytes)
            ),
            "portable_checkpoint_bytes": checkpoint_bytes,
        },
        "checkpoint": {
            "write_seconds_mean": mean_write,
            "read_seconds_mean": mean_read,
            "write_bytes_per_second": checkpoint_bytes / mean_write,
            "read_bytes_per_second": checkpoint_bytes / mean_read,
            "exact_roundtrip": checkpoint_success,
            "parent_checkpoint_id": restart.parent_checkpoint_id,
        },
        "physics": {
            "poisson_relative_residual": float(poisson.relative_residual),
            "potential_zero_mode_absolute": float(poisson.zero_mode_absolute),
            "accepted": bool(result.successful),
            "terminal_status": result.state.status,
            "stable_time_level": float(result.state.time),
        },
        "scaling": {
            "processes": 1,
            "devices": jax.device_count(),
            "serial_efficiency": 1.0,
        },
        "successful": successful,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n")
    if not successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
