# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Actual upstream accelerator identity and placement; never a CPU substitute."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.utils import canonicalize_name

from tools.check_accelerator_extras import Accelerator


RUNTIME_PROPERTY = "accelerator_runtime"


def selected_accelerator() -> Accelerator:
    value = os.environ.get("PHYDRAX_ACCELERATOR")
    if value is None:
        raise RuntimeError(
            "Set PHYDRAX_ACCELERATOR to explicitly select accelerator qualification"
        )
    return Accelerator(value)


def accelerator_device() -> Any:
    import jax

    from phydrax import config
    from phydrax.backends import oneapi_environment, tpu_environment

    selected = selected_accelerator()
    if config.snapshot().platforms != (selected.value,):
        raise RuntimeError(f"Qualification requires exclusive JAX_PLATFORMS={selected}")
    match selected:
        case Accelerator.ONEAPI:
            report = oneapi_environment(initialize_backend=True)
        case Accelerator.TPU:
            report = tpu_environment(initialize_backend=True)
    if not report.successful or not report.facts.device_probed:
        raise RuntimeError(f"Accelerator is unavailable: {report.issues}")
    devices = jax.local_devices(backend=selected.value)
    if not devices or jax.process_count() != 1:
        raise RuntimeError(
            "This bounded qualification requires one process and an actual local selected device"
        )
    if not config.x64_enabled():
        raise RuntimeError(
            "Qualification requires x64 for exact wide integer addressing, not a claim of FP64"
        )
    device = devices[0]
    runtime = str(device.client.platform_version)
    client_platform = str(device.client.platform)
    if selected is Accelerator.TPU:
        valid = device.platform == "tpu" and client_platform == "tpu"
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


def runtime_evidence(*arrays: Any) -> dict[str, object]:
    import jax

    device = accelerator_device()
    if not arrays:
        raise ValueError("Runtime evidence requires actual scientific result arrays")
    for array in arrays:
        if not isinstance(array, jax.Array) or array.devices() != {device}:
            raise RuntimeError(
                f"Scientific result is not placed exclusively on {device}: {array}"
            )
        array.block_until_ready()
    return {
        "selected": selected_accelerator().value,
        "platform": device.platform,
        "client_platform": device.client.platform,
        "platform_version": device.client.platform_version,
        "device_kind": device.device_kind,
        "qualified_device_id": device.id,
        "execution_device_count": 1,
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "global_device_count": jax.device_count(selected_accelerator().value),
        "local_device_count": jax.local_device_count(selected_accelerator().value),
        "x64_enabled": bool(jax.config.x64_enabled),
        "matmul_precision": str(jax.config.jax_default_matmul_precision),
    }


def record_execution(
    record_property: Callable[[str, object], None], *arrays: Any
) -> None:
    record_property(
        RUNTIME_PROPERTY, json.dumps(runtime_evidence(*arrays), sort_keys=True)
    )


def diagnostic_evidence() -> dict[str, Any]:
    """Retain provider-owned probed file/runtime facts without inventing build IDs."""
    from phydrax.backends import oneapi_environment, tpu_environment

    match selected_accelerator():
        case Accelerator.ONEAPI:
            report = oneapi_environment(initialize_backend=True)
        case Accelerator.TPU:
            report = tpu_environment(initialize_backend=True)
    if not report.successful or not report.facts.device_probed:
        raise RuntimeError(f"Selected runtime diagnostics refused: {report.issues}")
    return asdict(report)


def installed_provenance(
    venv: Path, expected_core: Mapping[str, str]
) -> dict[str, object]:
    """Bind actual installed bytes, including upstream RECORDs, to this fresh venv."""
    import phydrax

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
    for name in sorted(names | {"phydrax", *selected.distributions}):
        distribution = metadata.distribution(name)
        expected_version = selected.distributions.get(name)
        if expected_version is not None and distribution.version != expected_version:
            raise RuntimeError(f"{name} version differs from selected upstream stack")
        if not distribution.files:
            raise RuntimeError(f"{name} has no installed RECORD")
        files: dict[str, str] = {}
        for member in sorted(distribution.files, key=str):
            path = Path(str(distribution.locate_file(member))).resolve()
            if not path.is_relative_to(venv.resolve()):
                raise RuntimeError(
                    f"{name} contains a file outside the fresh environment: {path}"
                )
            if path.suffix == ".pyc":
                continue
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").digest()
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
    return {
        "package": str(package),
        "venv": str(venv.resolve()),
        "distributions": distributions,
    }
