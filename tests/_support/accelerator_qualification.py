# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Actual upstream accelerator identity, placement, and compiled-region evidence.

Never a CPU substitute: scientific results must live on the selected device.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any, TYPE_CHECKING

from packaging.utils import canonicalize_name

from tools.check_accelerator_extras import Accelerator


if TYPE_CHECKING:
    import jax


RUNTIME_PROPERTY = "accelerator_runtime"
RESOURCE_PROPERTY = "accelerator_resources"
EVIDENCE_NONCE_VARIABLE = "PHYDRAX_QUALIFICATION_NONCE"
WARM_REPEATS = 3
_SAMPLE_INTERVAL_SECONDS = 0.01


def selected_accelerator() -> Accelerator:
    value = os.environ.get("PHYDRAX_ACCELERATOR")
    if value is None:
        raise RuntimeError(
            "Set PHYDRAX_ACCELERATOR to explicitly select accelerator qualification"
        )
    return Accelerator(value)


def accelerator_device() -> jax.Device:
    import jax

    from phydrax import config
    from phydrax.backends import oneapi_environment, rocm_environment, tpu_environment

    selected = selected_accelerator()
    if config.snapshot().platforms != selected.qualification_platforms:
        raise RuntimeError(
            "Qualification requires "
            f"JAX_PLATFORMS={','.join(selected.qualification_platforms)}"
        )
    match selected:
        case Accelerator.ONEAPI:
            report = oneapi_environment(initialize_backend=True)
        case Accelerator.TPU:
            report = tpu_environment(initialize_backend=True)
        case Accelerator.ROCM10:
            report = rocm_environment(initialize_backend=True)
    if not report.successful or not report.facts.device_probed:
        raise RuntimeError(f"Accelerator is unavailable: {report.issues}")
    devices = jax.local_devices(backend=selected.runtime_platform)
    if not devices or jax.process_count() != 1:
        raise RuntimeError(
            "This bounded qualification requires one process and an actual local selected device"
        )
    if not config.x64_enabled():
        raise RuntimeError(
            "Qualification requires x64 for exact wide integer addressing, not a claim of FP64"
        )
    device_id = int(os.environ.get("PHYDRAX_ACCELERATOR_DEVICE_ID", "0"))
    matching = [device for device in devices if device.id == device_id]
    if len(matching) != 1:
        raise RuntimeError(f"Selected device ID {device_id} is not a unique local device")
    device = matching[0]
    runtime = str(device.client.platform_version)
    client_platform = str(device.client.platform)
    if selected is Accelerator.TPU:
        valid = device.platform == "tpu" and client_platform == "tpu"
    elif selected.rocm_major is not None:
        valid = selected.matches_rocm_runtime(device.platform, client_platform, runtime)
    else:
        valid = device.platform == "oneapi" or (
            device.platform == "gpu"
            and (
                client_platform.lower() in {"oneapi", "sycl"}
                or runtime.lower().startswith(("oneapi", "sycl"))
            )
        )
    if not valid:
        raise RuntimeError(
            f"Selected {selected} but actual runtime is {device.platform}/{client_platform}/{runtime}"
        )
    return device


def runtime_evidence(*arrays: object) -> dict[str, object]:
    import jax

    from phydrax import config
    from phydrax.backends import frontend_index_patch_identity

    device = accelerator_device()
    if not arrays:
        raise ValueError("Runtime evidence requires actual scientific result arrays")
    for array in arrays:
        if not isinstance(array, jax.Array) or array.devices() != {device}:
            raise RuntimeError(
                f"Scientific result is not placed exclusively on {device}: {array}"
            )
        array.block_until_ready()
    settings = config.snapshot()
    return {
        "selected": selected_accelerator().value,
        "rocm_major": selected_accelerator().rocm_major,
        "platform": device.platform,
        "client_platform": device.client.platform,
        "platform_version": device.client.platform_version,
        "device_kind": device.device_kind,
        "qualified_device_id": device.id,
        "execution_device_count": 1,
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "global_device_count": jax.device_count(selected_accelerator().runtime_platform),
        "local_device_count": jax.local_device_count(
            selected_accelerator().runtime_platform
        ),
        "x64_enabled": settings.enable_x64,
        "matmul_precision": str(settings.matmul_precision),
        "frontend_index_patch": frontend_index_patch_identity(),
    }


def record_execution(
    record_property: Callable[[str, object], None], *arrays: object
) -> None:
    record_property(
        RUNTIME_PROPERTY, json.dumps(runtime_evidence(*arrays), sort_keys=True)
    )


def _signature(value: object) -> list[dict[str, object]]:
    import jax

    return [
        {
            "shape": list(leaf.shape),
            "dtype": str(leaf.dtype),
            "weak_type": bool(jax.typeof(leaf).weak_type),
        }
        for leaf in jax.tree.leaves(value)
    ]


def compiled_region_evidence(
    function: Callable[..., Any],
    arguments: Sequence[Any],
    /,
    *,
    device: jax.Device,
    warm_repeats: int = WARM_REPEATS,
) -> tuple[Any, dict[str, object]]:
    """Lower, compile, and execute one scientific region on ``device`` by phase.

    Lowering and compilation are host phases. First and warm executions are
    synchronized and memory-sampled separately; the returned value is the last
    warm result of this measured executable. XLA's compiled memory plan is
    ``None`` with a reason when the provider exposes none, and allocator
    counters remain explicit unavailable measurements; neither becomes zero.
    """
    import equinox as eqx
    import jax

    from benchmarks._runtime import (
        logical_array_bytes,
        measure_host,
        measure_repeated,
        measure_synchronized,
    )
    from phydrax.execution import compiled_memory_estimate, PhaseMemorySampler

    placed = jax.device_put(tuple(arguments), device)
    statics: list[Any] = []

    def arrays_only(*values: jax.Array) -> Any:
        dynamic, static = eqx.partition(function(*values), eqx.is_array)
        statics.append(static)
        return dynamic

    lowered, lowering_seconds = measure_host(lambda: jax.jit(arrays_only).lower(*placed))
    compiled, compilation_seconds = measure_host(lowered.compile)

    def execute() -> Any:
        return compiled(*placed)

    with PhaseMemorySampler(
        "first-execution", interval_seconds=_SAMPLE_INTERVAL_SECONDS, devices=(device,)
    ) as first:
        _, first_seconds = measure_synchronized(execute)
    with PhaseMemorySampler(
        "warm-execution", interval_seconds=_SAMPLE_INTERVAL_SECONDS, devices=(device,)
    ) as warm:
        dynamic, warm_seconds = measure_repeated(execute, warmup=0, repeats=warm_repeats)
    estimate = compiled_memory_estimate(compiled)
    return eqx.combine(dynamic, statics[0]), {
        "program_sha256": hashlib.sha256(lowered.as_text().encode()).hexdigest(),
        "input_signature": _signature(placed),
        "output_signature": _signature(dynamic),
        "lowering_seconds": lowering_seconds,
        "compilation_seconds": compilation_seconds,
        "first_execution_seconds": first_seconds,
        "warm_execution": warm_seconds.to_seconds_dict(),
        "logical_retained_bytes": logical_array_bytes(dynamic),
        "compiled_memory": None if estimate is None else estimate.to_payload(),
        "compiled_memory_unavailable_reason": (
            "The provider exposes no XLA compiled memory analysis"
            if estimate is None
            else None
        ),
        "phase_memory": {
            "first_execution": first.evidence.to_payload(),
            "warm_execution": warm.evidence.to_payload(),
        },
    }


def record_compiled_region(
    record_property: Callable[[str, object], None],
    region: str,
    function: Callable[..., Any],
    /,
    *arguments: Any,
) -> Any:
    """Measure one region on the selected device and bind it to this run."""
    import equinox as eqx
    import jax

    outputs, evidence = compiled_region_evidence(
        function, arguments, device=accelerator_device()
    )
    record_property(
        RESOURCE_PROPERTY,
        json.dumps(
            {
                "region": region,
                "evidence_nonce": os.environ.get(EVIDENCE_NONCE_VARIABLE),
                "runtime": runtime_evidence(
                    *jax.tree.leaves(eqx.filter(outputs, eqx.is_array))
                ),
                **evidence,
            },
            sort_keys=True,
        ),
    )
    return outputs


def diagnostic_evidence() -> dict[str, object]:
    """Retain provider-owned probed file/runtime facts without inventing build IDs."""
    from phydrax.backends import oneapi_environment, rocm_environment, tpu_environment

    match selected_accelerator():
        case Accelerator.ONEAPI:
            report = oneapi_environment(initialize_backend=True)
        case Accelerator.TPU:
            report = tpu_environment(initialize_backend=True)
        case Accelerator.ROCM10:
            report = rocm_environment(initialize_backend=True)
    if not report.successful or not report.facts.device_probed:
        raise RuntimeError(f"Selected runtime diagnostics refused: {report.issues}")
    evidence = asdict(report)
    if selected_accelerator() is Accelerator.ROCM10:
        evidence["automatic_runtime_environment"] = {
            key: os.environ.get(key)
            for key in (
                "ROCM_PATH",
                "LLVM_PATH",
                "HIP_DEVICE_LIB_PATH",
                "LD_LIBRARY_PATH",
            )
        }
    return evidence


def _installed_path(distribution: metadata.Distribution, member: str, venv: Path) -> Path:
    located = distribution.locate_file(member)
    if not isinstance(located, Path):
        raise RuntimeError(f"Qualification requires filesystem-installed files: {member}")
    path = located.resolve()
    if not path.is_relative_to(venv.resolve()):
        raise RuntimeError(f"Installed file is outside the fresh environment: {path}")
    return path


def installed_provenance(
    venv: Path, expected_core: Mapping[str, str]
) -> dict[str, object]:
    """Bind actual installed bytes, including upstream RECORDs, to this fresh venv."""
    import phydrax
    from phydrax.backends import frontend_index_patch_identity

    selected = selected_accelerator()
    package = Path(phydrax.__file__).resolve()
    if not package.is_relative_to(venv.resolve()):
        raise RuntimeError(
            f"Source checkout imported instead of installed wheel: {package}"
        )
    distributions: dict[str, object] = {}
    # Bind transitive SDK/math libraries too, not just the two PJRT entry wheels.
    names = {
        canonicalize_name(str(item.metadata["Name"])) for item in metadata.distributions()
    }
    if selected is Accelerator.ROCM10 and any(
        name.startswith("jax-rocm") and name not in selected.distributions
        for name in names
    ):
        raise RuntimeError("ROCm qualification refuses unsupported runtime families")
    for name in sorted(names | {"phydrax", *selected.distributions}):
        distribution = metadata.distribution(name)
        expected_version = selected.distributions.get(name)
        if expected_version is not None and distribution.version != expected_version:
            raise RuntimeError(f"{name} version differs from selected upstream stack")
        if not distribution.files:
            raise RuntimeError(f"{name} has no installed RECORD")
        files: dict[str, str] = {}
        for member in sorted(distribution.files, key=str):
            path = _installed_path(distribution, str(member), venv)
            if path.suffix == ".pyc":
                continue
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").digest()
            if (
                name in selected.runtime_distributions
                and not str(member).endswith(".dist-info/RECORD")
                and member.hash is None
            ):
                raise RuntimeError(
                    f"Owned runtime file has no RECORD hash: {name}/{member}"
                )
            if member.hash is not None:
                if (
                    member.hash.mode != "sha256"
                    or base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
                    != member.hash.value
                ):
                    raise RuntimeError(
                        f"Installed upstream RECORD mismatch: {name}/{member}"
                    )
            files[str(member)] = digest.hex()
        if name == "phydrax":
            for relative, digest in expected_core.items():
                if files.get(relative) != digest:
                    raise RuntimeError(
                        f"Installed core differs from supplied wheel: {relative}"
                    )
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        distributions[name] = {
            "version": distribution.version,
            "files": files,
            "installed_sha256": hashlib.sha256(canonical).hexdigest(),
        }
    bundle_evidence = None
    if selected is Accelerator.ROCM10:
        from importlib import import_module

        bundle = import_module("phydrax_rocm_runtime._loader").inspect_rocm_bundle()
        if bundle is None:
            raise RuntimeError("Owned ROCm runtime bundle is missing")
        manifest_path = Path(bundle.manifest_path).resolve()
        if not manifest_path.is_relative_to(venv.resolve()):
            raise RuntimeError("ROCm bundle manifest leaks an external SDK path")
        payloads: dict[str, dict[str, str]] = {}
        for name in selected.runtime_distributions:
            if name == "phydrax-rocm-runtime":
                continue
            distribution = metadata.distribution(name)
            inventories = [
                member
                for member in distribution.files or ()
                if str(member).endswith(".dist-info/phydrax-payload.json")
            ]
            if len(inventories) != 1:
                raise RuntimeError(f"{name} has no unique payload inventory")
            payloads[name] = {
                "path": str(inventories[0]),
                "text": _installed_path(
                    distribution, str(inventories[0]), venv
                ).read_text(),
            }
            for item in json.loads(payloads[name]["text"]):
                path = _installed_path(distribution, item["path"], venv)
                status = path.stat()
                if (
                    status.st_size != item["size"]
                    or status.st_mode & 0o7777 != item["mode"]
                ):
                    raise RuntimeError(
                        f"Owned runtime payload size/mode mismatch: {path}"
                    )
        bundle_evidence = {"manifest": manifest_path.read_text(), "payloads": payloads}
    return {
        "package": str(package),
        "venv": str(venv.resolve()),
        "distributions": distributions,
        "frontend_index_patch": frontend_index_patch_identity(),
        "rocm_bundle": bundle_evidence,
    }
