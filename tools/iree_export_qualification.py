#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/iree_export.json")
    )
    args = parser.parse_args()
    matrix = jnp.asarray(((1.0, -0.5), (0.25, 2.0)), dtype=jnp.float32)

    def model(value, *, key=None):
        del key
        return jnp.tanh(value @ matrix)

    sample = jnp.asarray(((0.2, -0.3), (1.0, 0.5)), dtype=jnp.float32)
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "model.phxiree"
        started = time.perf_counter()
        exported = phx.export.save_iree(
            model,
            destination,
            inputs=(sample,),
            input_names=("x",),
            validate=True,
        )
        compile_seconds = time.perf_counter() - started
        executable = phx.export.load_iree(destination)
        started = time.perf_counter()
        deployed = executable(np.asarray(sample))
        warm_seconds = time.perf_counter() - started
        native = np.asarray(model(sample))
        payload = {
            "artifact_bytes": (destination / exported.manifest.module_file)
            .stat()
            .st_size,
            "compiler_version": exported.manifest.compiler_version,
            "runtime_version": exported.manifest.runtime_version,
            "target_backend": exported.manifest.target_backend,
            "runtime_driver": exported.manifest.runtime_driver,
            "compile_seconds": compile_seconds,
            "warm_seconds": warm_seconds,
            "maximum_absolute_error": float(np.max(np.abs(native - deployed))),
            "maximum_relative_error": exported.manifest.maximum_relative_error,
            "validation_ok": exported.manifest.validation_ok,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
