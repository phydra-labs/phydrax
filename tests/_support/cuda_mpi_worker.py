"""Real rank-local CUDA MPI qualification phases, launched in isolated processes."""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import sys
import traceback
import uuid
from collections.abc import Callable


for _dependency in ("mpi4jax", "mpi4py"):
    if importlib.util.find_spec(_dependency) is None:
        print(
            f"Optional CUDA MPI provider unavailable: {_dependency} is not installed",
            flush=True,
        )
        sys.exit(77)

from mpi4py import MPI


# This provider owns rank-local MPI, not JAX multi-host initialization.
for _key in ("SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE"):
    os.environ.pop(_key, None)

import jax
import numpy as np

from phydrax._execution_device import device_platform
from phydrax.backends.distributed import mpi4jax_availability, Mpi4JaxCollectiveProvider


type Operation = Callable[[jax.Array], jax.Array]
type Differential = Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]


def skip(reason: str) -> None:
    print(f"Optional CUDA MPI qualification unavailable: {reason}", flush=True)
    sys.exit(77)


def cuda_uuid(device: jax.Device) -> str:
    """Resolve the actual CUDA-visible ordinal through the initialized driver."""
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuDeviceGet.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.c_int)
    driver.cuDeviceGet.restype = ctypes.c_int
    driver.cuDeviceGetUuid_v2.argtypes = (ctypes.c_void_p, ctypes.c_int)
    driver.cuDeviceGetUuid_v2.restype = ctypes.c_int
    handle = ctypes.c_int()
    status = driver.cuDeviceGet(ctypes.byref(handle), device.local_hardware_id)
    assert status == 0, ("cuDeviceGet", status)
    raw_uuid = ctypes.create_string_buffer(16)
    status = driver.cuDeviceGetUuid_v2(raw_uuid, handle)
    assert status == 0, ("cuDeviceGetUuid_v2", status)
    return "GPU-" + str(uuid.UUID(bytes=raw_uuid.raw))


def assert_result(
    result: jax.Array, expected: np.ndarray, device: jax.Device
) -> list[float]:
    result.block_until_ready()
    assert result.devices() == {device}, result.devices()
    assert result.dtype == expected.dtype, (result.dtype, expected.dtype)
    np.testing.assert_array_equal(np.asarray(result), expected)
    return np.asarray(result).tolist()


def input_values(rank: int, step: int, dtype: np.dtype) -> np.ndarray:
    # Sub-float32 increments exercise genuine float64 arithmetic when selected.
    base = np.asarray([rank + 1 + 2.0**-30, 4 - 2 * rank - 2.0**-29], dtype=dtype)
    delta = np.asarray([0.25 * (rank + 1), -0.5 * (rank + 1)], dtype=dtype)
    return base + np.asarray(step, dtype=dtype) * delta


def operations(provider: Mpi4JaxCollectiveProvider, rank: int) -> dict[str, Operation]:
    def exchange(value: jax.Array) -> jax.Array:
        return provider.sendrecv(
            value, source=1 - rank, destination=1 - rank, sendtag=19, recvtag=19
        )

    return {
        "sum": provider.sum,
        "mean": provider.mean,
        "maximum": provider.maximum,
        "minimum": provider.minimum,
        "sendrecv": exchange,
    }


def primal_phase(
    provider: Mpi4JaxCollectiveProvider,
    rank: int,
    device: jax.Device,
    dtype: np.dtype,
    first_step: int,
) -> list[dict[str, object]]:
    compiled = {
        name: jax.jit(operation) for name, operation in operations(provider, rank).items()
    }
    records: list[dict[str, object]] = []
    # Reuse each executable with changed inputs; synchronize every result before
    # advancing to the next MPI call or issuing control-communicator traffic.
    for step in range(first_step, first_step + 3):
        inputs = [input_values(peer, step, dtype) for peer in range(2)]
        value = jax.device_put(inputs[rank], device)
        assert value.devices() == {device}
        total = inputs[0] + inputs[1]
        expected = {
            "sum": total,
            "mean": total / np.asarray(2, dtype=dtype),
            "maximum": np.maximum(*inputs),
            "minimum": np.minimum(*inputs),
            "sendrecv": inputs[1 - rank],
        }
        results = {
            name: assert_result(operation(value), expected[name], device)
            for name, operation in compiled.items()
        }
        jax.effects_barrier()
        records.append({"step": step, "values": results})
    return records


def jvp_operation(operation: Operation) -> Differential:
    def apply(value: jax.Array, tangent: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.jvp(operation, (value,), (tangent,))

    return apply


def vjp_operation(operation: Operation) -> Differential:
    def apply(value: jax.Array, cotangent: jax.Array) -> tuple[jax.Array, jax.Array]:
        primal, pullback = jax.vjp(operation, value)
        (adjoint,) = pullback(cotangent)
        return primal, adjoint

    return apply


def differentiation_phase(
    provider: Mpi4JaxCollectiveProvider,
    rank: int,
    device: jax.Device,
    dtype: np.dtype,
) -> dict[str, object]:
    routes = operations(provider, rank)
    inputs = [input_values(peer, 2, dtype) for peer in range(2)]
    seeds = [np.asarray([peer + 2, 2 * peer + 1], dtype=dtype) for peer in range(2)]
    value = jax.device_put(inputs[rank], device)
    seed = jax.device_put(seeds[rank], device)
    total = inputs[0] + inputs[1]
    expected_primals = {
        "sum": total,
        "mean": total / np.asarray(2, dtype=dtype),
        "sendrecv": inputs[1 - rank],
    }
    results: dict[str, object] = {}
    for name in ("sum", "mean"):
        primal, tangent = jax.jit(jvp_operation(routes[name]))(value, seed)
        expected = seeds[0] + seeds[1]
        if name == "mean":
            expected = expected / np.asarray(2, dtype=dtype)
        assert_result(primal, expected_primals[name], device)
        results[f"{name}.jvp"] = assert_result(tangent, expected, device)
        jax.effects_barrier()
    for name in ("sum", "mean", "sendrecv"):
        primal, adjoint = jax.jit(vjp_operation(routes[name]))(value, seed)
        # mpi4jax's SUM transpose is rank-local identity, not another reduction.
        expected = seeds[1 - rank] if name == "sendrecv" else seeds[rank]
        if name == "mean":
            expected = expected / np.asarray(2, dtype=dtype)
        assert_result(primal, expected_primals[name], device)
        results[f"{name}.vjp"] = assert_result(adjoint, expected, device)
        jax.effects_barrier()
    return results


def refusal_phase(
    provider: Mpi4JaxCollectiveProvider,
    rank: int,
    device: jax.Device,
    dtype: np.dtype,
) -> dict[str, object]:
    routes = operations(provider, rank)
    value = jax.device_put(input_values(rank, 0, dtype), device)
    tangent = jax.device_put(np.asarray([1, 2], dtype=dtype), device)
    refusals: dict[str, object] = {}
    for name, transform, transform_name, error_type in (
        ("maximum", jvp_operation, "jvp", NotImplementedError),
        ("minimum", jvp_operation, "jvp", NotImplementedError),
        ("maximum", vjp_operation, "vjp", NotImplementedError),
        ("minimum", vjp_operation, "vjp", NotImplementedError),
        ("sendrecv", jvp_operation, "jvp", RuntimeError),
    ):
        # Compile the whole transform so refusal occurs before any primal MPI
        # execution. The upstream sendrecv message reverses JVP/VJP terminology;
        # record it, but assert the actual unsupported transform and error type.
        transformed = jax.jit(transform(routes[name]))
        try:
            jax.block_until_ready(transformed(value, tangent))
        except error_type as error:
            refusals[f"{name}.{transform_name}"] = {
                "exception": type(error).__name__,
                "reason": str(error),
            }
        else:
            raise AssertionError(f"{name}.{transform_name} unexpectedly succeeded")
    jax.effects_barrier()
    return refusals


def lifecycle_phases(
    control: MPI.Intracomm, device: jax.Device, dtype: np.dtype, direct: bool
) -> list[dict[str, object]]:
    rank = control.Get_rank()
    records: list[dict[str, object]] = []
    for generation in range(2):
        # Never mix mpi4py control traffic with deferred device traffic.
        communicator = control.Dup()
        provider = Mpi4JaxCollectiveProvider(
            communicator, f"cuda-mpi-test-{generation}", gpu_direct=direct
        )
        assert provider.communicator is communicator
        record: dict[str, object] = {
            "generation": generation,
            "route": provider.provider_id,
            "primals": primal_phase(provider, rank, device, dtype, generation * 3),
        }
        if generation == 0:
            record["differentiation"] = differentiation_phase(
                provider, rank, device, dtype
            )
            record["refusals"] = refusal_phase(provider, rank, device, dtype)
        jax.effects_barrier()
        control.Barrier()
        del provider
        # Provider destruction must not free the caller-owned communicator.
        assert communicator.Get_rank() == rank
        assert communicator.Get_size() == 2
        communicator.Free()
        assert communicator == MPI.COMM_NULL
        record["caller_released"] = True
        assert control.allgather(generation) == [generation, generation]
        records.append(record)
    return records


def main() -> None:
    mode, placement, dtype_name = sys.argv[1:]
    direct = os.environ["MPI4JAX_USE_CUDA_MPI"] == "1"
    availability = mpi4jax_availability()
    if not availability.available:
        skip(availability.reason)
    cuda_devices = [
        device for device in jax.devices() if device_platform(device) == "cuda"
    ]
    required = 2 if placement == "distinct-gpus" else 1
    if len(cuda_devices) < required:
        skip(
            f"{placement} requires {required} visible CUDA devices; found {len(cuda_devices)}"
        )
    dtype = np.dtype(dtype_name)
    assert jax.config.x64_enabled, (
        "JAX_ENABLE_X64=true is required for this qualification"
    )
    control = MPI.COMM_WORLD.Dup()
    if mode == "probe":
        communicator = control.Dup()
        try:
            provider = Mpi4JaxCollectiveProvider(
                communicator, "cuda-mpi-probe", gpu_direct=direct
            )
        except RuntimeError as error:
            if str(error) not in (
                "mpi4jax was built without CUDA support",
                "MPI exposes no CUDA-awareness qualification query",
                "gpu_direct requires CUDA-aware MPI",
            ):
                raise
            skip(str(error))
        import mpi4jax

        if not mpi4jax.has_cuda_support():
            skip(
                "mpi4jax was built without CUDA support (also required for host-staged GPU input)"
            )
        del provider
        communicator.Free()
        control.Free()
        print(
            f"CUDA MPI {('direct' if direct else 'host-staged')} prerequisites available",
            flush=True,
        )
        return
    assert mode == "run", mode
    assert control.Get_size() == 2
    shared = control.Split_type(MPI.COMM_TYPE_SHARED)
    shared_rank_count = shared.Get_size()
    shared.Free()
    assert shared_rank_count == 2, shared_rank_count
    rank = control.Get_rank()
    device = cuda_devices[rank if placement == "distinct-gpus" else 0]
    ownership = control.allgather((MPI.Get_processor_name(), cuda_uuid(device)))
    assert len({physical_uuid for _, physical_uuid in ownership}) == required, ownership
    phases = lifecycle_phases(control, device, dtype, direct)
    control.Free()
    print(
        json.dumps(
            {
                "rank": rank,
                "dtype": dtype_name,
                "placement": placement,
                "gpu_direct": direct,
                "device_ownership": ownership,
                "local_hardware_id": device.local_hardware_id,
                "phases": phases,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
