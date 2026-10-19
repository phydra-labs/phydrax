# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Child scenarios exercising record transport and process isolation."""

import os
import sys

import jax


def record_probe(value: str) -> dict[str, str | int]:
    """Return topology and an argument while emitting non-JSON diagnostics."""
    print("[Gloo] Rank 0 is connected to 0 peer ranks")
    print("child diagnostic", file=sys.stderr)
    return {
        "value": value,
        "devices": jax.device_count(),
        "platform": jax.default_backend(),
        "extra_env": os.environ["PHYDRAX_TEST_RUNNER_TOKEN"],
    }


def fail_with_diagnostics() -> dict[str, str]:
    """Refuse after emitting both child diagnostic streams."""
    print("runner stdout evidence", flush=True)
    print("runner stderr evidence", file=sys.stderr, flush=True)
    raise RuntimeError("runner deliberate child refusal")
