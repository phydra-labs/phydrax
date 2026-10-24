# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Isolated physical CUDA probes; importing this module never discovers devices."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING, TypedDict

import numpy as np
from numpy.typing import NDArray


if TYPE_CHECKING:
    from phydrax_cuda13_pjrt.runtime import CudaRuntimeAPI


SIZES = (4, 4096, 1048580)
STEPS = (0, 1, 2)
MAX_BYTES = 16 * 1024 * 1024
CONSUMER_SIZES = (4, 12, 16, 40, 1024)
CONSUMER_ROUTES = ("peer-placement", "forced-copy", "host-control")


class Ownership(TypedDict):
    rank: int
    local_ordinal: int
    gpu_uuid: str
    platform: str


class CopyEvidence(TypedDict):
    route: str
    source_uuid: str
    destination_uuid: str
    bytes: int
    step: int
    expected_sha256: str
    observed_sha256: str
    mismatch_count: int
    ingress_sha256: str
    ingress_mismatch_count: int
    expected_prefix: list[int]
    ingress_prefix: list[int]
    observed_prefix: list[int]
    failure_stage: Literal["none", "source-ingress", "peer-transfer"]
    passed: bool


def checked(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed with CUDA status {status}")


def cuda_runtime() -> ctypes.CDLL | CudaRuntimeAPI:
    """Use only the CUDA runtime admitted by the canonical companion loader."""
    if sys.platform == "win32":
        from phydrax_cuda13_pjrt.runtime import cuda_runtime_api, library_handle

        from phydrax._cuda_native import load_cuda_component

        component = load_cuda_component("cudss")
        return cuda_runtime_api(library_handle(component.path))
    else:
        from phydrax._cuda_native import load_cuda_component
        from phydrax._elf_dependencies import read_dynamic_section

        component = load_cuda_component("cudss")
        paths = [
            path
            for path in component.preloaded
            if read_dynamic_section(path).soname == "libcudart.so.13"
        ]
        if len(paths) != 1:
            raise RuntimeError(f"Expected one audited CUDA13 runtime, got {paths}")
        runtime = ctypes.CDLL(paths[0])
    pointer, integer = ctypes.c_void_p, ctypes.c_int
    signatures: dict[str, tuple[Any, ...]] = {
        "cudaSetDevice": (integer,),
        "cudaDriverGetVersion": (ctypes.POINTER(integer),),
        "cudaRuntimeGetVersion": (ctypes.POINTER(integer),),
        "cudaMemGetInfo": (
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ),
        "cudaDeviceGetAttribute": (ctypes.POINTER(integer), integer, integer),
        "cudaDeviceCanAccessPeer": (ctypes.POINTER(integer), integer, integer),
        "cudaMalloc": (ctypes.POINTER(pointer), ctypes.c_size_t),
        "cudaFree": (pointer,),
        "cudaMemcpy": (pointer, pointer, ctypes.c_size_t, integer),
        "cudaMemcpyPeer": (pointer, integer, pointer, integer, ctypes.c_size_t),
        "cudaDeviceSynchronize": (),
        "cudaGetDriverEntryPointByVersion": (
            ctypes.c_char_p,
            ctypes.POINTER(pointer),
            ctypes.c_uint,
            ctypes.c_ulonglong,
            ctypes.POINTER(integer),
        ),
    }
    for name, arguments in signatures.items():
        function = getattr(runtime, name)
        function.argtypes = arguments
        function.restype = integer
    return runtime


def _driver_function(
    runtime: ctypes.CDLL | CudaRuntimeAPI, symbol: bytes, arguments: tuple[Any, ...]
) -> Any:
    pointer, query = ctypes.c_void_p(), ctypes.c_int()
    checked(
        runtime.cudaGetDriverEntryPointByVersion(
            symbol, ctypes.byref(pointer), 13000, 0, ctypes.byref(query)
        ),
        "cudaGetDriverEntryPointByVersion",
    )
    if query.value != 0 or pointer.value is None:
        raise RuntimeError(
            f"CUDA13 driver entry point unavailable: {symbol!r}, query={query.value}"
        )
    convention = ctypes.WINFUNCTYPE if sys.platform == "win32" else ctypes.CFUNCTYPE
    return convention(ctypes.c_int, *arguments)(pointer.value)


def ownership(runtime: ctypes.CDLL | CudaRuntimeAPI) -> list[Ownership]:
    import jax

    devices = jax.local_devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise ValueError("Physical CUDA qualification refuses CPU topology")
    result: list[Ownership] = []
    get_device = _driver_function(
        runtime, b"cuDeviceGet", (ctypes.POINTER(ctypes.c_int), ctypes.c_int)
    )
    get_uuid = _driver_function(
        runtime, b"cuDeviceGetUuid", (ctypes.c_void_p, ctypes.c_int)
    )
    for device in devices:
        raw = ctypes.create_string_buffer(16)
        handle = ctypes.c_int()
        checked(get_device(ctypes.byref(handle), device.local_hardware_id), "cuDeviceGet")
        checked(get_uuid(raw, handle), "cuDeviceGetUuid")
        result.append(
            {
                "rank": jax.process_index(),
                "local_ordinal": device.local_hardware_id,
                "gpu_uuid": "GPU-" + str(uuid.UUID(bytes=raw.raw)),
                "platform": device.platform,
            }
        )
    if len({row["gpu_uuid"] for row in result}) != len(result):
        raise ValueError("Duplicate physical GPU UUIDs")
    return result


def inventory(runtime: ctypes.CDLL | CudaRuntimeAPI) -> dict[str, Any]:
    from importlib.metadata import version as distribution_version

    import jax
    import jaxlib

    from benchmarks._runtime import capture_environment
    from phydrax._cuda_native import load_cuda_component

    component = load_cuda_component("cudss")
    devices = ownership(runtime)
    driver, version = ctypes.c_int(), ctypes.c_int()
    checked(runtime.cudaDriverGetVersion(ctypes.byref(driver)), "cudaDriverGetVersion")
    checked(runtime.cudaRuntimeGetVersion(ctypes.byref(version)), "cudaRuntimeGetVersion")
    if sys.platform == "win32":
        from phydrax_cuda13_pjrt.runtime import runtime_identity

        if component.major != 13 or version.value < 13000 or driver.value < 13000:
            raise ValueError(
                "Qualification requires CUDA13 and a compatible display driver"
            )
        identity = runtime_identity()
        runtime_evidence = {
            "compiled_cuda_major": component.major,
            "driver_compatibility_version": driver.value,
            "runtime_identity": identity._asdict(),
            "runtime_path": identity.runtime_path,
            "runtime_api_owner_path": runtime.owner_path,
        }
    else:
        if version.value // 1000 != 13:
            raise ValueError("Qualification requires the wheel-owned CUDA13 runtime")
        runtime_evidence = {
            "driver_version": driver.value,
            "runtime_path": str(runtime._name),
        }
    details = []
    for row in devices:
        ordinal = row["local_ordinal"]
        checked(runtime.cudaSetDevice(ordinal), "cudaSetDevice")
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        checked(
            runtime.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)),
            "cudaMemGetInfo",
        )
        major, minor = ctypes.c_int(), ctypes.c_int()
        checked(
            runtime.cudaDeviceGetAttribute(ctypes.byref(major), 75, ordinal),
            "compute major",
        )
        checked(
            runtime.cudaDeviceGetAttribute(ctypes.byref(minor), 76, ordinal),
            "compute minor",
        )
        details.append(
            dict(
                row,
                compute_capability=[major.value, minor.value],
                free_bytes_sample=free.value,
                total_bytes=total.value,
            )
        )
    connectivity = []
    for source in devices:
        for destination in devices:
            if source == destination:
                continue
            access = ctypes.c_int()
            checked(
                runtime.cudaDeviceCanAccessPeer(
                    ctypes.byref(access),
                    source["local_ordinal"],
                    destination["local_ordinal"],
                ),
                "cudaDeviceCanAccessPeer",
            )
            connectivity.append(
                {
                    "source_uuid": source["gpu_uuid"],
                    "destination_uuid": destination["gpu_uuid"],
                    "capability_reported": bool(access.value),
                }
            )
    return {
        "ownership": devices,
        "devices": details,
        **runtime_evidence,
        "environment": capture_environment().to_dict(),
        "cuda_distributions": {
            name: distribution_version(name)
            for name in (
                ("phydrax-cuda13-pjrt", "phydrax-cuda13", "nvidia-cuda-runtime")
                if sys.platform == "win32"
                else ("jax-cuda13-plugin", "jax-cuda13-pjrt", "nvidia-cuda-runtime")
            )
        },
        "native_component": {
            "name": component.name,
            "path": component.path,
            "version": component.version,
            "build_hash": component.build_hash,
            "major": component.major,
        },
        "runtime_version": version.value,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "peer_connectivity": connectivity,
        "peer_matrix_status": "not-applicable" if len(devices) == 1 else "required",
    }


def pattern(source: int, destination: int, size: int, step: int) -> NDArray[np.uint32]:
    if size <= 0 or size > MAX_BYTES or size % 4:
        raise ValueError("Copy sizes must be positive uint32-aligned bytes within 16 MiB")
    seed = np.uint32(
        (source + 1) * 104729 + (destination + 1) * 13007 + (step + 1) * 8191
    )
    return np.arange(size // 4, dtype=np.uint32) * np.uint32(2654435761) ^ seed


def _copy_evidence(
    expected: NDArray[np.uint32],
    ingress: NDArray[np.uint32],
    observed: NDArray[np.uint32],
    *,
    route: str,
    source_uuid: str,
    destination_uuid: str,
    step: int,
    expected_sha256: str,
) -> CopyEvidence:
    ingress_mismatches = int(np.count_nonzero(ingress != expected))
    mismatches = int(np.count_nonzero(observed != expected))
    stage: Literal["none", "source-ingress", "peer-transfer"] = "none"
    if ingress_mismatches:
        stage = "source-ingress"
    elif mismatches:
        stage = "peer-transfer"
    return {
        "route": route,
        "source_uuid": source_uuid,
        "destination_uuid": destination_uuid,
        "bytes": expected.nbytes,
        "step": step,
        "expected_sha256": expected_sha256,
        "observed_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
        "ingress_sha256": hashlib.sha256(ingress.tobytes()).hexdigest(),
        "ingress_mismatch_count": ingress_mismatches,
        "mismatch_count": mismatches,
        "expected_prefix": [int(value) for value in expected[:8]],
        "ingress_prefix": [int(value) for value in ingress[:8]],
        "observed_prefix": [int(value) for value in observed[:8]],
        "failure_stage": stage,
        "passed": stage == "none",
    }


def peer_copies(runtime: ctypes.CDLL | CudaRuntimeAPI, rows: list[CopyEvidence]) -> None:
    import jax

    devices = jax.local_devices()
    identities = ownership(runtime)
    for source, source_device in enumerate(devices):
        for destination, destination_device in enumerate(devices):
            if source == destination:
                continue
            src = source_device.local_hardware_id
            dst = destination_device.local_hardware_id
            for size in SIZES:
                source_pointer, destination_pointer = ctypes.c_void_p(), ctypes.c_void_p()
                try:
                    checked(runtime.cudaSetDevice(src), "cudaSetDevice(source)")
                    checked(
                        runtime.cudaMalloc(ctypes.byref(source_pointer), size),
                        "cudaMalloc(source)",
                    )
                    checked(runtime.cudaSetDevice(dst), "cudaSetDevice(destination)")
                    checked(
                        runtime.cudaMalloc(ctypes.byref(destination_pointer), size),
                        "cudaMalloc(destination)",
                    )
                    for step in STEPS:
                        expected = pattern(source, destination, size, step)
                        digest = hashlib.sha256(expected.tobytes()).hexdigest()
                        checked(runtime.cudaSetDevice(src), "cudaSetDevice(source)")
                        checked(
                            runtime.cudaMemcpy(
                                source_pointer, expected.ctypes.data, size, 1
                            ),
                            "cudaMemcpy(H2D)",
                        )
                        checked(
                            runtime.cudaDeviceSynchronize(),
                            "cudaDeviceSynchronize(source)",
                        )
                        source_observed = np.empty_like(expected)
                        checked(
                            runtime.cudaMemcpy(
                                source_observed.ctypes.data, source_pointer, size, 2
                            ),
                            "cudaMemcpy(source-local D2H)",
                        )
                        placed = jax.device_put(expected, source_device)
                        placed.block_until_ready()
                        if placed.devices() != {source_device}:
                            raise ValueError(
                                "JAX ingress did not reside on the requested source"
                            )
                        jax_source_observed = np.asarray(placed)
                        checked(runtime.cudaSetDevice(dst), "cudaSetDevice(destination)")
                        checked(
                            runtime.cudaMemcpyPeer(
                                destination_pointer, dst, source_pointer, src, size
                            ),
                            "cudaMemcpyPeer",
                        )
                        checked(
                            runtime.cudaDeviceSynchronize(),
                            "cudaDeviceSynchronize(destination)",
                        )
                        observed = np.empty_like(expected)
                        checked(
                            runtime.cudaMemcpy(
                                observed.ctypes.data, destination_pointer, size, 2
                            ),
                            "cudaMemcpy(D2H)",
                        )
                        copied = jax.device_put(placed, destination_device)
                        copied.block_until_ready()
                        if copied.devices() != {destination_device}:
                            raise ValueError(
                                "JAX copy did not reside on the requested destination"
                            )
                        for route, ingress, actual in (
                            ("cuda", source_observed, observed),
                            ("jax", jax_source_observed, np.asarray(copied)),
                        ):
                            rows.append(
                                _copy_evidence(
                                    expected,
                                    ingress,
                                    actual,
                                    route=route,
                                    source_uuid=identities[source]["gpu_uuid"],
                                    destination_uuid=identities[destination]["gpu_uuid"],
                                    step=step,
                                    expected_sha256=digest,
                                )
                            )
                finally:
                    if source_pointer.value is not None:
                        checked(runtime.cudaSetDevice(src), "cudaSetDevice(free source)")
                        checked(runtime.cudaFree(source_pointer), "cudaFree(source)")
                    if destination_pointer.value is not None:
                        checked(
                            runtime.cudaSetDevice(dst), "cudaSetDevice(free destination)"
                        )
                        checked(
                            runtime.cudaFree(destination_pointer), "cudaFree(destination)"
                        )


def bare_ownership() -> list[Ownership]:
    """Resolve physical identities through the admitted CUDA runtime."""
    return ownership(cuda_runtime())


def consumer_oracle(
    source: int, destination: int, size: int, step: int
) -> NDArray[np.int32]:
    # Small signed values avoid overflow and expose stale inputs on every step.
    return (pattern(source, destination, size, step) % np.uint32(100003)).astype(np.int32)


def consumer_evidence(
    expected: NDArray[np.int32],
    ingress: NDArray[np.int32],
    transferred: NDArray[np.int32],
    consumed: NDArray[np.int32],
    transfer_mismatches: int,
    consumer_mismatches: int,
) -> dict[str, Any]:
    wanted = expected * np.int32(3) + np.int32(7)
    row: dict[str, Any] = {}
    for name, actual, reference in (
        ("ingress", ingress, expected),
        ("transfer", transferred, expected),
        ("consumer", consumed, wanted),
    ):
        row[name + "_sha256"] = hashlib.sha256(actual.tobytes()).hexdigest()
        row[name + "_prefix"] = actual[:8].tolist()
        row[name + "_mismatch_count"] = int(np.count_nonzero(actual != reference))
    for name, values in (("expected", expected), ("expected_consumer", wanted)):
        row[name + "_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
        row[name + "_prefix"] = values[:8].tolist()
    row["device_transfer_mismatch_count"] = transfer_mismatches
    row["device_consumer_mismatch_count"] = consumer_mismatches
    stage = "none"
    if row["ingress_mismatch_count"]:
        stage = "source-ingress"
    elif transfer_mismatches:
        stage = "peer-transfer"
    elif consumer_mismatches:
        stage = "destination-consumer"
    elif row["transfer_mismatch_count"] or row["consumer_mismatch_count"]:
        stage = "host-readback"
    row.update(failure_stage=stage, passed=stage == "none")
    return row


def jax_consumers(record: dict[str, Any], required_gpus: int) -> None:
    import jax
    import jax.numpy as jnp

    identities = bare_ownership()
    record["ownership"] = identities
    record["peer_matrix_status"] = "not-applicable" if required_gpus == 1 else "required"
    if len(identities) != required_gpus:
        raise ValueError("Visible physical GPUs differ from --required-gpus")
    devices = jax.local_devices()

    def consume(
        value: jax.Array, reference: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        result = value * jnp.int32(3) + jnp.int32(7)
        return (
            result,
            jnp.count_nonzero(value != reference),
            jnp.count_nonzero(result != reference * jnp.int32(3) + jnp.int32(7)),
        )

    for destination, dst in enumerate(devices):
        consumer = jax.jit(consume)
        for source, src in enumerate(devices):
            if source == destination:
                continue
            for size in CONSUMER_SIZES:
                for step in STEPS:
                    expected = consumer_oracle(source, destination, size, step)
                    for route in CONSUMER_ROUTES:
                        identity = dict(
                            route=route,
                            source_uuid=identities[source]["gpu_uuid"],
                            destination_uuid=identities[destination]["gpu_uuid"],
                            bytes=size,
                            step=step,
                        )
                        stage = "source-ingress"
                        try:
                            placed = jax.device_put(expected, src)
                            placed.block_until_ready()
                            if placed.devices() != {src}:
                                raise ValueError("Incorrect source placement")
                            ingress = np.asarray(placed)
                            stage = "peer-transfer"
                            if route == "peer-placement":
                                copied = jax.device_put(placed, dst)
                            elif route == "forced-copy":
                                copied = jax.device_put(placed, dst, may_alias=False)
                            else:
                                # This control is explicit; it never repairs a peer route.
                                copied = jax.device_put(np.array(ingress, copy=True), dst)
                            copied.block_until_ready()
                            if copied.devices() != {dst}:
                                raise ValueError("Incorrect destination placement")
                            stage = "destination-consumer"
                            reference = jax.device_put(expected, dst)
                            result, transfer_bad, consumer_bad = consumer(
                                copied, reference
                            )
                            result.block_until_ready()
                            if result.devices() != {dst}:
                                raise ValueError(
                                    "Consumer did not execute at destination"
                                )
                            stage = "host-readback"
                            row = consumer_evidence(
                                expected,
                                ingress,
                                np.asarray(copied),
                                np.asarray(result),
                                int(transfer_bad),
                                int(consumer_bad),
                            )
                            record["rows"].append(dict(identity, **row))
                        except Exception as error:
                            record["rows"].append(
                                dict(
                                    identity,
                                    passed=False,
                                    failure_stage=stage,
                                    error=f"{type(error).__name__}: {error}",
                                )
                            )
    count = (
        required_gpus
        * (required_gpus - 1)
        * len(CONSUMER_ROUTES)
        * len(CONSUMER_SIZES)
        * len(STEPS)
    )
    record["coverage"] = {
        "expected_consumer_observations": count,
        "recorded_consumer_observations": len(record["rows"]),
        "passed_consumer_observations": sum(
            row["passed"] is True for row in record["rows"]
        ),
    }
    record["passed"] = len(record["rows"]) == count and all(
        row["passed"] for row in record["rows"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("inventory", "peer-copies", "jax-consumers"), required=True
    )
    parser.add_argument("--required-gpus", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from tools._accelerator_process import write_json_evidence

    record: dict[str, Any] = {
        "case": args.case,
        "required_gpus": args.required_gpus,
        "sizes_bytes": list(SIZES),
        "steps": list(STEPS),
        "passed": False,
        "rows": [],
        "errors": [],
    }
    if args.case == "jax-consumers":
        record.update(sizes_bytes=list(CONSUMER_SIZES), routes=list(CONSUMER_ROUTES))
        try:
            jax_consumers(record, args.required_gpus)
        except Exception as error:
            record["errors"].append(f"{type(error).__name__}: {error}")
        write_json_evidence(args.output, record)
        if not record["passed"]:
            raise SystemExit(1)
        return
    try:
        runtime = cuda_runtime()
        record.update(inventory(runtime))
        if len(record["ownership"]) != args.required_gpus:
            raise ValueError("Visible physical GPUs differ from --required-gpus")
        if args.case == "peer-copies":
            peer_copies(runtime, record["rows"])
        expected_count = (
            args.required_gpus * (args.required_gpus - 1) * 2 * len(SIZES) * len(STEPS)
            if args.case == "peer-copies"
            else 0
        )
        record["passed"] = len(record["rows"]) == expected_count and all(
            row["passed"] for row in record["rows"]
        )
        if len(record["rows"]) != expected_count:
            record["errors"].append("Missing required peer-copy observations")
        for route, source in sorted(
            {
                (row["route"], row["source_uuid"])
                for row in record["rows"]
                if row["ingress_mismatch_count"] != 0
            }
        ):
            record["errors"].append(
                f"Incorrect source-local {route} ingress contents on {source}; "
                "outgoing peer integrity cannot be certified"
            )
        if any(row["failure_stage"] == "peer-transfer" for row in record["rows"]):
            record["errors"].append(
                "Incorrect peer-copy contents after correct source-local ingress"
            )
    except Exception as error:
        record["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        record["coverage"] = {
            "expected_copy_observations": (
                args.required_gpus
                * (args.required_gpus - 1)
                * 2
                * len(SIZES)
                * len(STEPS)
                if args.case == "peer-copies"
                else 0
            ),
            "recorded_copy_observations": len(record["rows"]),
            "recorded_ingress_observations": len(record["rows"]),
            "passed_ingress_observations": sum(
                row["ingress_mismatch_count"] == 0 for row in record["rows"]
            ),
            "passed_copy_observations": sum(row["passed"] for row in record["rows"]),
        }
        write_json_evidence(args.output, record)
    print(
        json.dumps(
            {"case": args.case, "passed": record["passed"], "output": str(args.output)}
        ),
        flush=True,
    )
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
