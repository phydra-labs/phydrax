# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Consumer-side identities and native lifecycle persistence for energy examples."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import jax
import numpy as np

from phydrax import lifecycle


def json_bytes(value):
    return json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()


def identity(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def execution_identity():
    root = Path(__file__).resolve().parents[2]
    paths = sorted(path.relative_to(root) for path in (root / "phydrax").rglob("*.py"))
    paths += sorted(
        path.relative_to(root) for path in (root / "examples" / "energy").glob("*.py")
    )
    paths += [
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("tools/energy_qualification.py"),
        Path("tools/building_energy_benchmarks.py"),
        Path("tests/interchange/data/energy_accumulator.c"),
        Path("tests/interchange/data/energy_accumulator.xml"),
    ]
    packages = {
        item.metadata["Name"]: item.version
        for item in importlib.metadata.distributions()
        if item.metadata.get("Name")
    }
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": dict(sorted(packages.items())),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "devices": [str(device) for device in jax.devices()],
    }
    return {
        "build_id": lifecycle.digest_paths(root, paths),
        "source_paths": [str(path) for path in paths],
        "environment_id": identity(environment),
        "environment": environment,
    }


def archive_workflow(directory, name, metrics, arrays, units, checkpoint, *, execution):
    """Persist actual arrays and a numeric checkpoint; verify exact bytes on reopen.

    Checkpoints contain physical restart coordinates, not serialized executable
    solver caches. The caller reconstructs the explicitly identified native model.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    run_id = identity({"name": name, "metrics": metrics, "execution": execution})
    payloads = {key: np.asarray(value) for key, value in arrays.items()}
    payloads["execution_record"] = np.frombuffer(
        json_bytes({"metrics": metrics, "execution": execution}), dtype=np.uint8
    )
    fields = [(key, key, units[key]) for key in arrays] + [
        ("execution_record", "execution_record", "UTF-8 JSON bytes")
    ]
    manifest = lifecycle.ResultManifest(
        name,
        run_id,
        fields,
        {key: lifecycle.payload_digest(value) for key, value in payloads.items()},
    )
    result = lifecycle.create(
        directory / f"{name}.result.zip", manifest=manifest, arrays=payloads
    )
    reopened = lifecycle.open(result.path)
    if result.archive_id != reopened.archive_id or any(
        not np.array_equal(value, reopened.arrays[key]) for key, value in payloads.items()
    ):
        raise RuntimeError("Lifecycle result reopen changed a physical payload.")
    checkpoint_arrays = {key: np.asarray(value) for key, value in checkpoint.items()}
    checkpoint_arrays["execution_record"] = payloads["execution_record"]
    checkpoint_id = lifecycle.collection_digest(checkpoint_arrays)
    shards = tuple(
        lifecycle.CheckpointShard(
            key, lifecycle.payload_digest(value), lifecycle.payload_byte_count(value)
        )
        for key, value in checkpoint_arrays.items()
    )
    checkpoint_manifest = lifecycle.CheckpointManifest(
        checkpoint_id,
        name,
        lifecycle.collection_digest(payloads),
        execution["build_id"] + ":" + execution["environment_id"],
        shards,
        complete=True,
    )
    saved = lifecycle.create(
        directory / f"{name}.checkpoint.zip",
        manifest=checkpoint_manifest,
        arrays=checkpoint_arrays,
    )
    restored = lifecycle.open(saved.path)
    if saved.archive_id != restored.archive_id or any(
        not np.array_equal(value, restored.arrays[key])
        for key, value in checkpoint_arrays.items()
    ):
        raise RuntimeError(
            "Lifecycle checkpoint reopen changed physical restart coordinates."
        )
    return {"result": reopened, "checkpoint": restored, "run_id": run_id}


def archive_metrics(archives):
    return {
        "result_archive_id": archives["result"].archive_id,
        "result_path": str(archives["result"].path),
        "checkpoint_archive_id": archives["checkpoint"].archive_id,
        "checkpoint_path": str(archives["checkpoint"].path),
    }
