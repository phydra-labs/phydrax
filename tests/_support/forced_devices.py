# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Run importable test scenarios in isolated CPU device topologies."""

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from phydrax import config as pxconfig


_ROOT = Path(__file__).resolve().parents[2]
_TIMEOUT_SECONDS = 300


def gloo_rank_environment(coordinator: str, rank: int) -> dict[str, str]:
    """Bootstrap variables for one rank of a two-process Gloo run.

    `import phydrax` consumes them before any backend use, so scenario modules
    may import phydrax at module level and never initialize distribution
    themselves.
    """
    return pxconfig.environment(
        pxconfig.RuntimeConfig(
            bootstrap=pxconfig.RuntimeBootstrap(
                coordinator_address=coordinator,
                num_processes=2,
                process_id=rank,
                local_device_ids=(0,),
            ),
            cpu_collectives_implementation="gloo",
        ),
        base={},
    )


def run_on_forced_devices(
    module: str,
    function: str,
    args: tuple[str, ...],
    *,
    devices: int,
    env: Mapping[str, str],
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Return a child scenario's record without parsing diagnostic stdout.

    Each invocation owns a temporary record file, including concurrent ranks.
    The topology is selected before the child imports JAX.
    """
    if devices < 1:
        raise ValueError(f"forced_devices.devices must be positive; got {devices}")
    environment = pxconfig.environment(
        pxconfig.RuntimeConfig(
            platforms=("cpu",), cpu_device_count=devices, enable_x64=True
        ),
        base={**os.environ, **env},
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(_ROOT), environment.get("PYTHONPATH", ""))
    )
    with tempfile.TemporaryDirectory(prefix="phydrax-test-record-") as directory:
        record_path = Path(directory) / "record.json"
        process = subprocess.Popen(
            (
                sys.executable,
                "-m",
                "tests._support.forced_devices",
                "--record",
                str(record_path),
                module,
                function,
                *args,
            ),
            cwd=_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        # A failed collective can strand its peers or a nested cold subprocess.
        # The runner owns their process group and its bounded shutdown.
        try:
            stdout, stderr = process.communicate(
                timeout=_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if process.returncode or timed_out or not record_path.is_file():
            print(stdout, end="")
            print(stderr, end="", file=sys.stderr)
            pytest.fail(
                f"Child scenario {module}.{function} exited {process.returncode}; "
                f"timed out: {timed_out}; record exists: {record_path.is_file()}",
                pytrace=False,
            )
        record = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(
            f"forced_devices.record must be a JSON object; got {type(record).__name__}"
        )
    return record


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("module")
    parser.add_argument("function")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    pytest.register_assert_rewrite(arguments.module)
    module = importlib.import_module(arguments.module)
    # Scenario names are supplied by test code, not by a persisted record.
    function = vars(module)[arguments.function]
    record = function(*arguments.args)
    arguments.record.write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    _main()
