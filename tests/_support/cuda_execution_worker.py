# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Isolated physical CUDA probes; importing this module never discovers devices."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray


SIZES = (4, 4096, 1048580)
STEPS = (0, 1, 2)
MAX_BYTES = 16 * 1024 * 1024


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


def cuda_runtime() -> ctypes.CDLL:
    """Use only the CUDA runtime admitted by the canonical companion loader."""
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
    runtime: ctypes.CDLL, symbol: bytes, arguments: tuple[Any, ...]
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
    return ctypes.CFUNCTYPE(ctypes.c_int, *arguments)(pointer.value)


def ownership(runtime: ctypes.CDLL) -> list[Ownership]:
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


def inventory(runtime: ctypes.CDLL) -> dict[str, Any]:
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
    if version.value // 1000 != 13:
        raise ValueError("Qualification requires the wheel-owned CUDA13 runtime")
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
        "driver_version": driver.value,
        "environment": capture_environment().to_dict(),
        "cuda_distributions": {
            name: distribution_version(name)
            for name in ("jax-cuda13-plugin", "jax-cuda13-pjrt", "nvidia-cuda-runtime")
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
        "runtime_path": str(runtime._name),
        "peer_connectivity": connectivity,
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


def peer_copies(runtime: ctypes.CDLL, rows: list[CopyEvidence]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("inventory", "peer-copies"), required=True)
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
