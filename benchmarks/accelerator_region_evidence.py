#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Report qualification compiled-region evidence for the float32 Poisson PINN step.

Measures the advertised example's Adam update through the same recorder used by
installed-wheel accelerator qualification, on one explicitly selected local
device: separate lowering, compilation, first and warm synchronized execution,
logical retained bytes, XLA's compiled memory plan, and per-phase allocator
samples. Output is an observation, neither accelerator qualification nor a
performance claim; a CPU run exercises the recording path only.

Run: JAX_PLATFORMS=cpu python -m benchmarks.accelerator_region_evidence --platform cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from benchmarks._runtime import capture_environment
from examples.accelerator_scientific_execution import poisson_training
from phydrax import config as pxconfig
from tests._support.accelerator_qualification import (
    compiled_region_evidence,
    WARM_REPEATS,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warm-repeats", type=int, default=WARM_REPEATS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warm_repeats < 1:
        parser.error("--warm-repeats must be positive")
    # Qualification keeps x64 on for exact addressing; data stays float32.
    pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
    matching = [
        device
        for device in jax.local_devices(backend=args.platform)
        if device.id == args.device_id
    ]
    if len(matching) != 1:
        parser.error(f"Device ID {args.device_id} is not a unique {args.platform} device")
    device = matching[0]
    with jax.default_device(device):
        training = poisson_training()
        (_, _, loss, _), evidence = compiled_region_evidence(
            training.step,
            (training.parameters, training.state),
            device=device,
            warm_repeats=args.warm_repeats,
        )
    record = {
        "scope": "observed compiled-region evidence; not hardware qualification or performance",
        "region": "pinn-training-step",
        "device": {
            "platform": device.platform,
            "client_platform": device.client.platform,
            "platform_version": device.client.platform_version,
            "device_kind": device.device_kind,
            "device_id": device.id,
        },
        "environment": capture_environment().to_dict(),
        "loss": float(loss),
        "loss_finite": bool(np.isfinite(loss)),
        **evidence,
    }
    text = json.dumps(record, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    return 0 if record["loss_finite"] and evidence["compiled_memory"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
