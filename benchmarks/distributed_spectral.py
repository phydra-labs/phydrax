#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated distributed full-complex spectral FFT benchmark.

One-device and forced-CPU campaigns are functional evidence only. Physical
slab and nondegenerate pencil campaigns retain observed hardware evidence.
Inputs are generated shard-locally, with two changed-input buffers regardless
of repeat count. A bounded NumPy oracle checks a small transform; full campaign
round-trip and adjoint checks use native distributed scalar reductions.
Multi-process runs write one ``.process-N`` output per rank so local allocator
and retained-buffer measurements remain attributable to every physical device.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, TypeAlias

import jax
import numpy as np
from jax.stages import Compiled

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax import config as pxconfig
from phydrax._execution_collectives import (
    CollectiveLease,
    CollectiveProgram,
    prepare_collective_program,
)
from phydrax._execution_sampling import sample_device_memory, sample_host_memory
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization.spectral import (
    DistributedSpectralExecutionPlan,
    SpectralMeshTopology,
    SpectralPrecisionPolicy,
    SpectralTransposeProvider,
)
from phydrax.typing import parse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
Campaign: TypeAlias = Literal[
    "one-device-functional",
    "forced-cpu-functional",
    "physical-slab",
    "physical-pencil",
]
Schedule: TypeAlias = Literal["slab", "pencil"]
Precision: TypeAlias = Literal["float32", "float64"]


def _shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "shape must be comma-separated integers"
        ) from error
    if not shape or any(length <= 0 for length in shape):
        raise argparse.ArgumentTypeError("shape needs positive dimensions")
    return shape


def _campaign_contract(
    campaign: Campaign, schedule: Schedule, mesh_shape: tuple[int, ...]
) -> None:
    device_count = int(np.prod(mesh_shape))
    forces_host_devices = pxconfig.snapshot().cpu_device_count is not None
    physical = campaign in ("physical-slab", "physical-pencil")
    if physical:
        if forces_host_devices:
            raise ValueError(
                "Physical campaigns reject forced host-device-count XLA flags."
            )
        if device_count < 2:
            raise ValueError("Physical campaigns require at least two selected devices.")
    if campaign == "one-device-functional" and device_count != 1:
        raise ValueError("one-device-functional requires a one-device mesh.")
    if campaign == "forced-cpu-functional" and not forces_host_devices:
        raise ValueError(
            "forced-cpu-functional requires explicit "
            "--xla_force_host_platform_device_count in XLA_FLAGS."
        )
    if campaign == "physical-slab" and schedule != "slab":
        raise ValueError("physical-slab requires the slab schedule.")
    if campaign == "physical-pencil" and schedule != "pencil":
        raise ValueError("physical-pencil requires the pencil schedule.")
    expected_rank = 2 if schedule == "pencil" else 1
    if len(mesh_shape) != expected_rank:
        raise ValueError(f"{schedule} requires a rank-{expected_rank} mesh shape.")
    if campaign == "physical-pencil" and any(size < 2 for size in mesh_shape):
        raise ValueError("physical-pencil requires two nondegenerate mesh axes.")

    devices = tuple(jax.devices())
    if campaign == "forced-cpu-functional":
        if any(device.platform != "cpu" for device in devices):
            raise ValueError(
                "forced-cpu-functional requires an all-CPU JAX device inventory."
            )
        if jax.process_count() != 1:
            raise ValueError(
                "forced-cpu-functional is process-local functional evidence only."
            )
    if device_count > len(devices):
        raise ValueError(
            "mesh_shape exceeds this process's visible JAX device inventory."
        )
    if physical and any(device.platform == "cpu" for device in devices[:device_count]):
        raise ValueError("Physical campaigns require accelerator devices.")
    if jax.process_count() > 1 and {
        device.process_index for device in devices[:device_count]
    } != set(range(jax.process_count())):
        raise ValueError("The selected spectral mesh must include every JAX process.")


def _input_block(
    shape: tuple[int, ...],
    index: tuple[slice, ...] | None,
    sample: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """Generate only one addressable shard, never a global campaign grid."""
    if index is None:
        raise ValueError("Spectral input generation requires an explicit shard index.")
    grid = np.zeros(
        tuple(len(range(*part.indices(n))) for n, part in zip(shape, index, strict=True)),
        dtype=np.float64,
    )
    stride = 1
    for axis in reversed(range(len(shape))):
        coordinates = np.arange(*index[axis].indices(shape[axis]), dtype=np.float64)
        broadcast = [1] * len(shape)
        broadcast[axis] = coordinates.size
        grid += coordinates.reshape(broadcast) * stride
        stride *= shape[axis]
    phase = (sample + 1.0) / math.prod(shape)
    return np.asarray(
        np.sin((grid + 1.0) * phase) + 1j * np.cos((grid + 0.5) * (phase + 0.125)),
        dtype=dtype,
    )


def _placed_input(
    plan: DistributedSpectralExecutionPlan, sample: int, *, modal: bool = False
) -> jax.Array:
    layout = plan.modal_layout if modal else plan.physical_layout
    return jax.make_array_from_callback(
        plan.spatial_shape,
        layout.sharding(plan.topology),
        lambda index: _input_block(
            plan.spatial_shape, index, sample, np.dtype(plan.precision.coefficient_dtype)
        ),
    )


def _distributed_defect(
    plan: DistributedSpectralExecutionPlan,
    observed: jax.Array,
    expected: jax.Array,
    *,
    modal: bool,
) -> float:
    representation = "modal" if modal else "physical"
    error = plan.diagnostics(observed - expected, representation=representation)
    reference = plan.diagnostics(expected, representation=representation)
    if not bool(error.finite) or not bool(reference.finite):
        return float("inf")
    return float(error.maximum_absolute) / max(float(reference.maximum_absolute), 1.0)


@jax.jit
def _to_modal(plan: DistributedSpectralExecutionPlan, values: jax.Array) -> jax.Array:
    return plan.to_modal(values)


@jax.jit
def _to_physical(plan: DistributedSpectralExecutionPlan, values: jax.Array) -> jax.Array:
    return plan.to_physical(values)


def _stage(
    lease: CollectiveLease | None,
    function: Callable[[DistributedSpectralExecutionPlan, jax.Array], jax.Array],
    plan: DistributedSpectralExecutionPlan,
    values: jax.Array,
) -> tuple[Compiled | CollectiveProgram[jax.Array], dict[str, float | None]]:
    if lease is None:
        staged = jax.jit(function)
        executable, timing = measure_lower_and_compile(
            lambda: staged.lower(plan, values), lambda lowered: lowered.compile()
        )
        return executable, {
            "program_staging_lowering_compilation_seconds": (
                timing.lowering_seconds + timing.compilation_seconds
            ),
            "lowering_seconds": timing.lowering_seconds,
            "compilation_seconds": timing.compilation_seconds,
        }
    program, seconds = measure_host(
        lambda: prepare_collective_program(function, plan, values, lease_key=lease.spec)
    )
    return program, {
        "program_staging_lowering_compilation_seconds": seconds,
        "lowering_seconds": None,
        "compilation_seconds": None,
    }


def _execute(
    lease: CollectiveLease | None,
    operation: Compiled | CollectiveProgram[jax.Array],
    plan: DistributedSpectralExecutionPlan,
    values: jax.Array,
) -> jax.Array:
    if isinstance(operation, CollectiveProgram):
        if lease is None:
            raise ValueError("A collective program requires its explicitly live lease.")
        return lease.invoke(operation, plan, values)
    if lease is not None:
        raise TypeError(
            "Collective execution requires an explicitly staged CollectiveProgram."
        )
    return operation(plan, values)


def _small_numpy_reference(
    plan: DistributedSpectralExecutionPlan,
    collective_lease: CollectiveLease | None = None,
) -> tuple[tuple[int, ...], float, dict[str, Any]]:
    # Every mesh factor divides each reference dimension, admitting either
    # transpose schedule without making the oracle grow with the campaign.
    multiple = math.lcm(*plan.topology.mesh_shape)
    shape = tuple(min(length, 4 * multiple) for length in plan.spatial_shape)
    if math.prod(shape) > 65536:
        raise ValueError("The bounded NumPy reference exceeds 65536 coefficients.")
    reference = DistributedSpectralExecutionPlan(
        plan.topology,
        shape,
        owner_id=plan.owner_id,
        precision=plan.precision,
        admitted_payload_shapes=((),),
        state_shape=(),
        schedule=plan.schedule,
        transpose_provider=plan.transpose_provider,
        collective_binding=plan.collective_binding,
        maximum_external_scratch_bytes=plan.maximum_external_scratch_bytes,
    ).prepare()
    host = _input_block(
        shape, (slice(None),) * len(shape), 0, np.dtype(plan.precision.coefficient_dtype)
    )
    expected = np.asarray(
        np.fft.fftn(host, norm="ortho") * reference.transform_scale,
        dtype=np.dtype(plan.precision.coefficient_dtype),
    )
    distributed_expected = jax.make_array_from_callback(
        shape,
        reference.modal_layout.sharding(reference.topology),
        lambda index: expected[index],
    )
    reference_input = _placed_input(reference, 0)
    operation, program_preparation = _stage(
        collective_lease, _to_modal, reference, reference_input
    )
    defect = _distributed_defect(
        reference,
        _execute(collective_lease, operation, reference, reference_input),
        distributed_expected,
        modal=True,
    )
    return (
        shape,
        defect,
        {
            "execution_id": reference.execution_id,
            "logical_resource_report": asdict(reference.report.resource),
            "program_preparation_timing": program_preparation,
            "transpose_preparation_evidence": (
                None
                if reference.report.transpose_evidence is None
                else asdict(reference.report.transpose_evidence)
            ),
        },
    )


def _hardware_evidence(topology: SpectralMeshTopology) -> dict[str, Any]:
    selected = tuple(topology.mesh.devices.flat)
    visible = tuple(jax.devices())
    return {
        "host": {
            "node": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "load_average_1_5_15": (
                None if sys.platform == "win32" else list(os.getloadavg())
            ),
            "load_average_unavailable_reason": (
                "Windows does not provide os.getloadavg"
                if sys.platform == "win32"
                else None
            ),
        },
        "process": {
            "index": jax.process_index(),
            "count": jax.process_count(),
            "local_device_count": jax.local_device_count(),
            "global_device_count": jax.device_count(),
        },
        "selected_devices": [
            {
                "id": int(device.id),
                "process_index": int(device.process_index),
                "platform": device.platform,
                "device_kind": device.device_kind,
            }
            for device in selected
        ],
        "visible_devices": [
            {
                "id": int(device.id),
                "process_index": int(device.process_index),
                "platform": device.platform,
                "device_kind": device.device_kind,
            }
            for device in visible
        ],
        "scope": "observed by this process at benchmark execution time",
    }


def _run(
    campaign: Campaign,
    schedule: Schedule,
    shape: tuple[int, ...],
    mesh_shape: tuple[int, ...],
    precision_name: Precision,
    repeats: int,
    maximum_bytes: int,
    *,
    transpose_provider: SpectralTransposeProvider = "native",
    collective_lease: CollectiveLease | None = None,
    maximum_external_scratch_bytes: int = 256 * 1024**2,
) -> dict[str, Any]:
    """Keep measurement/evidence in one scope so snapshots share buffer lifetimes."""
    if len(shape) < 2:
        raise ValueError("distributed C2C campaigns require spatial rank at least two.")
    _campaign_contract(campaign, schedule, mesh_shape)
    provider = parse(transpose_provider, SpectralTransposeProvider, "transpose_provider")
    if (provider == "cudecomp") != (collective_lease is not None):
        raise ValueError(
            "cuDecomp campaigns require one explicit live collective lease; native campaigns do not."
        )
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    precision = SpectralPrecisionPolicy(np.dtype(precision_name))
    topology = SpectralMeshTopology(mesh_shape)
    owner_id = canonical_fingerprint(
        {"kind": "distributed-spectral-benchmark-owner", "driver": Path(__file__).name}
    )
    plan, preparation_seconds = measure_host(
        lambda: DistributedSpectralExecutionPlan(
            topology,
            shape,
            owner_id=owner_id,
            precision=precision,
            admitted_payload_shapes=((),),
            state_shape=(),
            schedule=schedule,
            maximum_bytes=maximum_bytes,
            transpose_provider=provider,
            collective_binding=None
            if collective_lease is None
            else collective_lease.binding,
            maximum_external_scratch_bytes=maximum_external_scratch_bytes,
        ).prepare()
    )
    baseline_memory = {
        "host": sample_host_memory().to_payload(),
        "devices": [
            sample.to_payload() for sample in sample_device_memory(jax.local_devices())
        ],
    }
    placed, generation_placement_seconds = measure_synchronized(
        lambda: _placed_input(plan, 0)
    )
    warm_placed, warm_generation_placement_seconds = measure_synchronized(
        lambda: tuple(_placed_input(plan, sample) for sample in (1, 2))
    )
    placed_memory = {
        "host": sample_host_memory().to_payload(),
        "devices": [
            sample.to_payload() for sample in sample_device_memory(jax.local_devices())
        ],
    }
    forward_operation, compilation = _stage(collective_lease, _to_modal, plan, placed)
    placed_probe = _placed_input(plan, 3, modal=True)
    inverse_operation, inverse_compilation = _stage(
        collective_lease, _to_physical, plan, placed_probe
    )
    compiled = (
        forward_operation.executable
        if isinstance(forward_operation, CollectiveProgram)
        else forward_operation
    )
    compiled_memory = {
        "host": sample_host_memory().to_payload(),
        "devices": [
            sample.to_payload() for sample in sample_device_memory(jax.local_devices())
        ],
    }
    first, first_seconds = measure_synchronized(
        lambda: _execute(collective_lease, forward_operation, plan, placed)
    )
    warm_index = 0

    def changed_input_execution() -> jax.Array:
        nonlocal warm_index
        value = warm_placed[warm_index]
        warm_index = (warm_index + 1) % len(warm_placed)
        return _execute(collective_lease, forward_operation, plan, value)

    warm_result, warm = measure_repeated(
        changed_input_execution, warmup=0, repeats=repeats
    )
    cost = compiled.cost_analysis()
    compiler = compiler_evidence(
        cost,
        compiled.memory_analysis(),
        source="xla",
        opaque_custom_calls=provider == "cudecomp",
    )

    runtime_memory = {
        "host": sample_host_memory().to_payload(),
        "devices": [
            sample.to_payload() for sample in sample_device_memory(jax.local_devices())
        ],
    }
    # Only replicated scalar diagnostics cross to the host, including on
    # multi-process runs whose scientific arrays are not fully addressable.
    reference, reference_seconds = measure_host(
        lambda: _small_numpy_reference(plan, collective_lease)
    )
    reference_shape, numerical_defect, reference_evidence = reference
    round_trip_defect = _distributed_defect(
        plan,
        _execute(collective_lease, inverse_operation, plan, first),
        placed,
        modal=False,
    )
    lhs = complex(plan.global_inner_product(first, placed_probe))
    rhs = (
        complex(
            plan.global_inner_product(
                placed,
                _execute(collective_lease, inverse_operation, plan, placed_probe),
                representation="physical",
            )
        )
        * plan.transform_scale**2
    )
    adjoint_defect = float(abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0))
    tolerance = 5.0e-5 if precision_name == "float32" else 1.0e-11

    return {
        "campaign": campaign,
        "schedule": schedule,
        "transpose_provider": provider,
        "transpose_preparation_evidence": (
            None
            if plan.report.transpose_evidence is None
            else asdict(plan.report.transpose_evidence)
        ),
        "shape": list(shape),
        "mesh_shape": list(mesh_shape),
        "ids": {
            "owner_id": plan.owner_id,
            "numerical_id": plan.numerical_id,
            "execution_id": plan.execution_id,
            "plan_id": plan.plan_id,
            "topology_id": topology.topology_id,
            "physical_layout_id": plan.physical_layout.layout_id,
            "modal_layout_id": plan.modal_layout.layout_id,
            "forward_sequence_id": plan.report.forward_sequence_id,
            "inverse_sequence_id": plan.report.inverse_sequence_id,
            "padded_forward_sequence_id": plan.report.padded_forward_sequence_id,
            "padded_inverse_sequence_id": plan.report.padded_inverse_sequence_id,
            "precision_policy_id": precision.policy_id,
            "payload_admission_id": canonical_fingerprint(
                {
                    "kind": "distributed-spectral-payload-admission",
                    "shapes": [list(payload) for payload in plan.admitted_payload_shapes],
                }
            ),
            "resource_report_id": plan.report.resource.report_id,
        },
        "static_contract": {
            "admitted_payload_shapes": [
                list(payload) for payload in plan.admitted_payload_shapes
            ],
            "state_shape": list(plan.state_shape),
            "coefficient_dtype": precision.coefficient_dtype,
            "transform_dtype": precision.transform_dtype,
            "reduction_dtype": precision.reduction_dtype,
            "sequence_identity_scope": "core static stage sequences, also bound by execution_id",
            "host_gather": plan.report.host_gather,
            "collective_count": plan.report.collective_count,
        },
        "timing": {
            "plan_and_provider_preparation_seconds": preparation_seconds,
            "bounded_numpy_reference_seconds": reference_seconds,
            "generate_and_place_seconds": generation_placement_seconds,
            "warm_changed_input_generate_and_place_seconds": (
                warm_generation_placement_seconds
            ),
            **compilation,
            "inverse_program_preparation_timing": inverse_compilation,
            "program_preparation_scope": "explicit host staging, semantic agreement, lowering and compilation",
            "first_synchronized_seconds": first_seconds,
            "warm_changed_input": warm.to_seconds_dict(),
        },
        "memory": {
            "logical_input_bytes": logical_array_bytes(placed),
            "logical_output_bytes": logical_array_bytes(warm_result),
            "baseline_scope": "after plan and selected provider preparation",
            "baseline": baseline_memory,
            "after_warm_execution": runtime_memory,
            "after_placement": placed_memory,
            "after_compilation": compiled_memory,
            "after_validation": {
                "host": sample_host_memory().to_payload(),
                "devices": [
                    sample.to_payload()
                    for sample in sample_device_memory(jax.local_devices())
                ],
            },
            "retained_changed_input_count": len(warm_placed),
            "retained_changed_input_bytes": logical_array_bytes(warm_placed),
            "retained_modal_probe_bytes": logical_array_bytes(placed_probe),
            "scope": "process-local samples; allocator peaks are since process start",
            "compiler": {**asdict(compiler), "raw_cost_analysis": cost},
            "logical_resource_report": {
                "canonical_storage_bytes": plan.report.resource.canonical_storage_bytes,
                "padded_storage_bytes": plan.report.resource.padded_storage_bytes,
                "transform_workspace_bytes": plan.report.resource.transform_workspace_bytes,
                "collective_payload_bytes": plan.report.resource.collective_payload_bytes,
                "peak_live_bytes": plan.report.resource.peak_live_bytes,
                "maximum_bytes": plan.report.resource.maximum_bytes,
                "transpose_packing_bytes": plan.report.resource.transpose_packing_bytes,
                "external_device_scratch_bytes": plan.report.resource.external_device_scratch_bytes,
                "external_resident_device_bytes": plan.report.resource.external_resident_device_bytes,
                "external_host_bytes": plan.report.resource.external_host_bytes,
            },
        },
        "defects": {
            "bounded_numpy_forward_relative_linf": numerical_defect,
            "distributed_round_trip_relative_linf": round_trip_defect,
            "distributed_adjoint_relative": adjoint_defect,
        },
        "reference": {
            "shape": list(reference_shape),
            "maximum_coefficients": 65536,
            "campaign_payload_host_gather": False,
            "reduction_dtype": precision.reduction_dtype,
            **reference_evidence,
        },
        "passed": all(
            np.isfinite(defect) and defect <= tolerance
            for defect in (numerical_defect, round_trip_defect, adjoint_defect)
        ),
        "tolerance": tolerance,
        "hardware_process_host_evidence": _hardware_evidence(topology),
        "qualification": {
            "claimed": False,
            "scope": (
                "functional-only forced CPU or one-device evidence"
                if campaign in ("one-device-functional", "forced-cpu-functional")
                else "explicit physical campaign measurement; not qualification"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="one-device-functional")
    parser.add_argument("--schedule", default="slab")
    parser.add_argument("--shape", type=_shape, default=(32, 32, 32))
    parser.add_argument("--mesh-shape", type=_shape, default=(1,))
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--maximum-bytes",
        type=int,
        default=1 << 31,
        help="Native plan budget; extra benchmark buffers are reported separately.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "distributed_spectral.json",
    )
    arguments = parser.parse_args()
    campaign = parse(arguments.campaign, Campaign, "campaign")
    schedule = parse(arguments.schedule, Schedule, "schedule")
    precision = parse(arguments.precision, Precision, "precision")
    configuration = {
        "campaign": campaign,
        "schedule": schedule,
        "shape": list(arguments.shape),
        "mesh_shape": list(arguments.mesh_shape),
        "precision": precision,
        "repeats": arguments.repeats,
        "maximum_bytes": arguments.maximum_bytes,
    }
    row = _run(
        campaign,
        schedule,
        arguments.shape,
        arguments.mesh_shape,
        precision,
        arguments.repeats,
        arguments.maximum_bytes,
    )
    record: dict[str, Any] = {
        "configuration": configuration,
        "row": row,
        "environment": capture_environment().to_dict(),
    }
    record["identity"] = capture_benchmark_identity(
        PROJECT_ROOT, Path(__file__), tuple(record)
    ).to_dict()
    record["workload_id"] = canonical_fingerprint(
        {
            "kind": "distributed-spectral-benchmark-workload",
            "configuration": configuration,
        }
    )
    output = arguments.output
    if jax.process_count() > 1:
        output = output.with_name(
            f"{output.stem}.process-{jax.process_index()}{output.suffix}"
        )
    write_json_atomic(output, record)
    if not row["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
