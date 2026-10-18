#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Explicit N-rank, single-host CUDA-aware MPI measurements; no RDMA claim."""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import os
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Any, assert_never, TYPE_CHECKING


if TYPE_CHECKING:
    import jax
    import numpy as np
    from jax import Array
    from mpi4py.MPI import Comm

    from phydrax.backends.distributed import Mpi4JaxCollectiveProvider


class Route(StrEnum):
    CUDA_DIRECT = "cuda-direct"
    HOST_STAGED = "host-staged"


class Operation(StrEnum):
    SUM = "sum"
    SENDRECV = "sendrecv"


class Precision(StrEnum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIZES = (8, 1024, 65536, 1048576, 16777216, 67108864)


def _bootstrap(rank_count: int) -> tuple[Comm, Comm, dict[str, str]]:
    # MPI owns these ranks. Initialize it before disabling independent JAX bootstrap.
    from mpi4py import MPI

    keys = ("SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE")
    launch = {key: os.environ.pop(key) for key in keys if key in os.environ}
    if MPI.COMM_WORLD.Get_size() != rank_count:
        raise ValueError(f"CUDA MPI measurement requires exactly {rank_count} MPI ranks")
    control = MPI.COMM_WORLD.Dup()
    data = MPI.COMM_WORLD.Dup()
    return control, data, launch


def _gpu_uuid(device: jax.Device) -> str:
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuDeviceGetUuid_v2.argtypes = (ctypes.c_void_p, ctypes.c_int)
    driver.cuDeviceGetUuid_v2.restype = ctypes.c_int
    value = ctypes.create_string_buffer(16)
    status = driver.cuDeviceGetUuid_v2(value, device.local_hardware_id)
    if status != 0:
        raise RuntimeError(f"CUDA GPU UUID query failed with status {status}")
    return "GPU-" + str(uuid.UUID(bytes=value.raw))


def _dtype(precision: Precision) -> np.dtype:
    import numpy as np

    match precision:
        case Precision.FLOAT32:
            return np.dtype(np.float32)
        case Precision.FLOAT64:
            return np.dtype(np.float64)
        case _:
            assert_never(precision)


def _operation(
    provider: Mpi4JaxCollectiveProvider, operation: Operation, rank: int, rank_count: int
) -> Callable[[Array], Array]:
    match operation:
        case Operation.SUM:
            return provider.sum
        case Operation.SENDRECV:

            def exchange(value: Array) -> Array:
                return provider.sendrecv(
                    value,
                    source=(rank - 1) % rank_count,
                    destination=(rank + 1) % rank_count,
                )

            return exchange
        case _:
            assert_never(operation)


def _memory(device: jax.Device) -> dict[str, Any]:
    from phydrax.execution import sample_device_memory, sample_host_memory

    return {
        "host": sample_host_memory().to_payload(),
        "device": sample_device_memory((device,))[0].to_payload(),
    }


def _check_result(
    result: Array,
    base: np.ndarray,
    operation: Operation,
    rank: int,
    rank_count: int,
    delta: int,
    device: jax.Device,
) -> None:
    import numpy as np

    if result.devices() != {device} or result.dtype != base.dtype:
        raise RuntimeError("MPI changed the declared output device or dtype")
    match operation:
        case Operation.SUM:
            expected = np.zeros_like(base)
            for peer in range(rank_count):
                expected += base + np.asarray(peer + delta, dtype=base.dtype)
        case Operation.SENDRECV:
            expected = base + np.asarray(
                (rank - 1) % rank_count + delta, dtype=base.dtype
            )
        case _:
            assert_never(operation)
    # Independent bounded host reference, outside all timed communication regions.
    np.testing.assert_array_equal(np.asarray(result), expected)


def _profile(start: bool) -> None:
    driver = ctypes.CDLL("libcuda.so.1")
    operation = driver.cuProfilerStart if start else driver.cuProfilerStop
    operation.argtypes = ()
    operation.restype = ctypes.c_int
    status = operation()
    if status != 0:
        raise RuntimeError(f"CUDA profiler capture boundary failed with status {status}")


def _point(
    provider: Mpi4JaxCollectiveProvider,
    control: Comm,
    device: jax.Device,
    operation: Operation,
    precision: Precision,
    size_bytes: int,
    repeats: int,
    warmup: int,
    capture: bool,
) -> dict[str, Any]:
    import jax
    import numpy as np

    from benchmarks._runtime import (
        compiler_evidence,
        DurationDistribution,
        measure_host,
        measure_synchronized,
    )

    dtype = _dtype(precision)
    if size_bytes % dtype.itemsize:
        raise ValueError("Message sizes must be divisible by the declared dtype size")
    count = size_bytes // dtype.itemsize
    rank = control.Get_rank()
    rank_count = control.Get_size()

    def prepare() -> tuple[np.ndarray, tuple[Array, Array]]:
        base = (np.arange(count, dtype=np.int32) % np.int32(257)).astype(dtype)
        first = jax.device_put(base + np.asarray(rank, dtype=dtype), device)
        second = jax.device_put(base + np.asarray(rank + 8, dtype=dtype), device)
        return base, (first, second)

    (base, values), prepare_seconds = measure_synchronized(prepare)
    compiled_operation = jax.jit(_operation(provider, operation, rank, rank_count))
    lowered, lower_seconds = measure_host(lambda: compiled_operation.lower(values[0]))
    compiled, compile_seconds = measure_host(lowered.compile)
    # The JIT dispatcher owns ordered MPI effect tokens. JAX 0.11.2's AOT
    # Compiled.__call__ fast path loses that token after its first invocation.
    memory_before = _memory(device)
    control.Barrier()
    first, first_seconds = measure_synchronized(lambda: compiled_operation(values[0]))
    _check_result(first, base, operation, rank, rank_count, 0, device)
    del first
    control.Barrier()
    changed, changed_seconds = measure_synchronized(lambda: compiled_operation(values[1]))
    _check_result(changed, base, operation, rank, rank_count, 8, device)
    del changed
    for index in range(warmup):
        control.Barrier()
        compiled_operation(values[index % 2]).block_until_ready()
    samples = []
    if capture:
        _profile(True)
    result = None
    for index in range(repeats):
        # The control communicator never overlaps deferred device MPI operations.
        # Its barrier is deliberately outside the measured invocation.
        control.Barrier()
        result, elapsed = measure_synchronized(
            lambda: compiled_operation(values[index % 2])
        )
        samples.append(elapsed)
    if capture:
        _profile(False)
    if result is None:
        raise RuntimeError("A positive repeat count must produce an MPI result")
    memory_after = _memory(device)
    _check_result(
        result, base, operation, rank, rank_count, 8 * ((repeats - 1) % 2), device
    )
    distribution = DurationDistribution(tuple(samples))
    return {
        "operation": operation.value,
        "precision": precision.value,
        "message_bytes_per_rank": size_bytes,
        "prepare_and_place_seconds": prepare_seconds,
        "lowering_seconds": lower_seconds,
        "compilation_seconds": compile_seconds,
        "first_synchronized_seconds": first_seconds,
        "first_changed_input_seconds": changed_seconds,
        "warm": distribution.to_dict(),
        "logical_retained_input_bytes": sum(value.nbytes for value in values),
        "raw_compiler_cost_analysis": compiled.cost_analysis(),
        "compiler": asdict(
            compiler_evidence(
                None,
                compiled.memory_analysis(),
                source="rank-local XLA executable; excludes external MPI/UCX storage",
                unavailable_reason="Opaque MPI calls report unknown XLA costs (including negative sentinel values); raw analysis is retained separately.",
            )
        ),
        "memory_before_execution": memory_before,
        "memory_after_warm_execution": memory_after,
        "passed": True,
    }


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", type=Route, choices=tuple(Route), required=True)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument(
        "--precision",
        nargs="+",
        type=Precision,
        choices=tuple(Precision),
        default=tuple(Precision),
    )
    parser.add_argument(
        "--operation",
        nargs="+",
        type=Operation,
        choices=tuple(Operation),
        default=tuple(Operation),
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Mark one warmed point for a separate CUDA-profiler capture, never performance evidence.",
    )
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--maximum-buffer-bytes", type=int, default=67108864)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.ranks < 2:
        parser.error("ranks must be at least two")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup nonnegative")
    if not 1 <= args.maximum_buffer_bytes <= 2147483647:
        parser.error("maximum buffer bytes must be in [1, 2147483647]")
    if any(
        size < 8 or size > args.maximum_buffer_bytes or size % 8 for size in args.sizes
    ):
        parser.error("sizes must be multiples of eight within the explicit buffer budget")
    if args.capture and (
        len(args.sizes) != 1 or len(args.precision) != 1 or len(args.operation) != 1
    ):
        parser.error(
            "profiler capture requires exactly one size, precision, and operation"
        )
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("message sizes must not repeat")
    if len(set(args.precision)) != len(args.precision) or len(set(args.operation)) != len(
        args.operation
    ):
        parser.error("precisions and operations must not repeat")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    control, data, launch = _bootstrap(args.ranks)
    import jax
    from mpi4py import MPI

    from benchmarks._io import write_json_atomic
    from benchmarks._runtime import capture_benchmark_identity, capture_environment
    from phydrax.backends.distributed import Mpi4JaxCollectiveProvider
    from phydrax.typing import parse

    route = parse(args.route, Route, "route")
    precisions = tuple(parse(value, Precision, "precision") for value in args.precision)
    operations = tuple(parse(value, Operation, "operation") for value in args.operation)
    rank = control.Get_rank()
    devices = jax.devices("gpu")
    if len(devices) != args.ranks or jax.process_count() != 1:
        raise RuntimeError(
            f"Each MPI rank must see exactly {args.ranks} GPUs without JAX multi-host bootstrap"
        )
    device = devices[rank]
    shared = control.Split_type(MPI.COMM_TYPE_SHARED)
    shared_rank_count = shared.Get_size()
    shared.Free()
    ownership = control.allgather(
        {
            "rank": rank,
            "host": MPI.Get_processor_name(),
            "gpu_uuid": _gpu_uuid(device),
            "hardware_id": device.local_hardware_id,
            "kind": device.device_kind,
        }
    )
    if (
        shared_rank_count != args.ranks
        or len({item["gpu_uuid"] for item in ownership}) != args.ranks
        or len({item["host"] for item in ownership}) != 1
        or [item["rank"] for item in ownership] != list(range(args.ranks))
    ):
        raise RuntimeError(
            f"Qualification requires one shared-memory domain and {args.ranks} distinct physical GPUs"
        )
    provider = Mpi4JaxCollectiveProvider(
        data, "cuda-mpi-measurement", gpu_direct=route is Route.CUDA_DIRECT
    )
    rows = []
    for precision in precisions:
        for operation in operations:
            for size in sorted(args.sizes):
                rows.append(
                    _point(
                        provider,
                        control,
                        device,
                        operation,
                        precision,
                        size,
                        args.repeats,
                        args.warmup,
                        args.capture,
                    )
                )
    jax.effects_barrier()
    records = control.gather(
        {
            "rank": rank,
            "environment": capture_environment().to_dict(),
            "rows": rows,
        },
        root=0,
    )
    del provider
    data.Free()
    control.Free()
    if rank == 0:
        identity = capture_benchmark_identity(
            ROOT,
            Path(__file__),
            ("configuration", "mpi", "ownership", "ranks", "scope", "passed"),
        )
        record = {
            "identity": identity.to_dict(),
            "configuration": {
                "route": route.value,
                "ranks": args.ranks,
                "sizes": sorted(args.sizes),
                "precision": [value.value for value in precisions],
                "operation": [value.value for value in operations],
                "repeats": args.repeats,
                "warmup": args.warmup,
                "maximum_buffer_bytes": args.maximum_buffer_bytes,
                "profile_capture": args.capture,
            },
            "mpi": {
                "library": MPI.Get_library_version(),
                "mpi4py": importlib.metadata.version("mpi4py"),
                "mpi4jax": importlib.metadata.version("mpi4jax"),
                "launch_environment": launch,
                "transport_environment": {
                    key: value
                    for key, value in sorted(os.environ.items())
                    if key.startswith(("UCX_", "OMPI_MCA_", "MPI4JAX_"))
                },
            },
            "ownership": ownership,
            "ranks": records,
            "scope": {
                "hosts": 1,
                "mpi_ranks": args.ranks,
                "shared_memory_ranks": shared_rank_count,
                "distinct_gpus": args.ranks,
                "cuda_aware": route is Route.CUDA_DIRECT,
                "mpi4jax_host_staging": route is Route.HOST_STAGED,
                "zero_host_copy_claim": False,
                "network_rdma_claim": False,
                "timing": "Local synchronized duration; control barriers excluded. Aggregate corresponding rank samples using their maximum.",
                "memory": "Process-local allocator counters; peaks accumulate across the ordered message-size sweep.",
            },
            "passed": True,
        }
        write_json_atomic(args.output, record)
        print(f"CUDA MPI measurement passed: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
