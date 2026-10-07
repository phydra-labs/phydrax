#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated distributed full-complex spectral FFT benchmark.

One-device and forced-CPU campaigns are functional evidence only. Physical
slab and pencil campaigns retain exact observed hardware, process, host,
topology, and layout evidence; selecting such a campaign does not assert that
its measurements qualify any machine or release.
"""

from __future__ import annotations

import argparse
import os
import platform
import shlex
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, TypeAlias

import jax
import numpy as np

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
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization.spectral import (
    DistributedSpectralExecutionPlan,
    SpectralMeshTopology,
    SpectralPrecisionPolicy,
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


_FORCED_HOST_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"


def _has_forced_host_device_count(flags: str, /) -> bool:
    tokens = tuple(shlex.split(flags))
    return any(
        token == _FORCED_HOST_DEVICE_COUNT_FLAG
        or token.startswith(f"{_FORCED_HOST_DEVICE_COUNT_FLAG}=")
        for token in tokens
    )


def _campaign_contract(
    campaign: Campaign, schedule: Schedule, mesh_shape: tuple[int, ...]
) -> None:
    device_count = int(np.prod(mesh_shape))
    flags = os.environ.get("XLA_FLAGS", "")
    forces_host_devices = _has_forced_host_device_count(flags)
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


def _host_inputs(
    shape: tuple[int, ...], count: int, dtype: np.dtype[Any]
) -> tuple[np.ndarray, ...]:
    grid = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
    denominator = float(max(grid.size, 1))
    values = []
    for sample in range(count):
        phase = (sample + 1.0) / denominator
        real = np.sin((grid + 1.0) * phase)
        imaginary = np.cos((grid + 0.5) * (phase + 0.125))
        values.append(np.asarray(real + 1j * imaginary, dtype=dtype))
    return tuple(values)


def _relative_defect(observed: np.ndarray, expected: np.ndarray) -> float:
    numerator = float(np.max(np.abs(observed - expected), initial=0.0))
    denominator = max(float(np.max(np.abs(expected), initial=0.0)), 1.0)
    return numerator / denominator


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
            "load_average_1_5_15": list(os.getloadavg()),
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
) -> dict[str, Any]:
    if len(shape) < 2:
        raise ValueError("distributed C2C campaigns require spatial rank at least two.")
    _campaign_contract(campaign, schedule, mesh_shape)
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    precision = SpectralPrecisionPolicy(np.dtype(precision_name))
    topology = SpectralMeshTopology(mesh_shape)
    owner_id = canonical_fingerprint(
        {"kind": "distributed-spectral-benchmark-owner", "driver": Path(__file__).name}
    )
    plan = DistributedSpectralExecutionPlan(
        topology,
        shape,
        owner_id=owner_id,
        precision=precision,
        admitted_payload_shapes=((),),
        state_shape=(),
        schedule=schedule,
    ).prepare()
    host_values, host_preparation_seconds = measure_host(
        lambda: _host_inputs(shape, repeats + 3, np.dtype(precision.coefficient_dtype))
    )
    placed, placement_seconds = measure_synchronized(
        lambda: plan.place(host_values[0], representation="physical")
    )
    warm_placed, warm_placement_seconds = measure_synchronized(
        lambda: tuple(
            plan.place(value, representation="physical")
            for value in host_values[1 : repeats + 1]
        )
    )
    transformed = jax.jit(plan.to_modal)
    compiled, compilation = measure_lower_and_compile(
        lambda: transformed.lower(placed), lambda lowered: lowered.compile()
    )
    first, first_seconds = measure_synchronized(lambda: compiled(placed))
    warm_index = 0

    def changed_input_execution() -> jax.Array:
        nonlocal warm_index
        value = warm_placed[warm_index]
        warm_index = (warm_index + 1) % len(warm_placed)
        return compiled(value)

    warm_result, warm = measure_repeated(
        changed_input_execution, warmup=0, repeats=repeats
    )
    compiler = compiler_evidence(
        compiled.cost_analysis(), compiled.memory_analysis(), source="xla"
    )

    # These checks are intentionally independent of the timed executable path:
    # NumPy supplies the full-complex oracle and complex128 host accumulations.
    first_host = np.asarray(first)
    expected = np.fft.fftn(host_values[0], norm="ortho") * plan.transform_scale
    numerical_defect = _relative_defect(first_host, expected)
    round_trip = np.asarray(plan.to_physical(first))
    round_trip_defect = _relative_defect(round_trip, host_values[0])
    adjoint_probe = host_values[repeats + 1]
    placed_probe = plan.place(adjoint_probe, representation="modal")
    inverse_probe = np.asarray(plan.to_physical(placed_probe))
    lhs = np.vdot(first_host.astype(np.complex128), adjoint_probe.astype(np.complex128))
    rhs = np.vdot(
        host_values[0].astype(np.complex128),
        (plan.transform_scale * plan.transform_scale * inverse_probe).astype(
            np.complex128
        ),
    )
    adjoint_defect = float(abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0))

    return {
        "campaign": campaign,
        "schedule": schedule,
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
            "host_preparation_seconds": host_preparation_seconds,
            "placement_seconds": placement_seconds,
            "warm_changed_input_placement_seconds": warm_placement_seconds,
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
            "first_synchronized_seconds": first_seconds,
            "warm_changed_input": warm.to_seconds_dict(),
        },
        "memory": {
            "logical_input_bytes": logical_array_bytes(placed),
            "logical_output_bytes": logical_array_bytes(warm_result),
            "compiler": asdict(compiler),
            "logical_resource_report": {
                "canonical_storage_bytes": plan.report.resource.canonical_storage_bytes,
                "padded_storage_bytes": plan.report.resource.padded_storage_bytes,
                "transform_workspace_bytes": plan.report.resource.transform_workspace_bytes,
                "collective_payload_bytes": plan.report.resource.collective_payload_bytes,
                "peak_live_bytes": plan.report.resource.peak_live_bytes,
                "maximum_bytes": plan.report.resource.maximum_bytes,
            },
        },
        "defects": {
            "independent_numpy_forward_relative_linf": numerical_defect,
            "independent_numpy_round_trip_relative_linf": round_trip_defect,
            "independent_complex128_adjoint_relative": adjoint_defect,
        },
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
    }
    row = _run(
        campaign,
        schedule,
        arguments.shape,
        arguments.mesh_shape,
        precision,
        arguments.repeats,
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
    write_json_atomic(arguments.output, record)


if __name__ == "__main__":
    main()
