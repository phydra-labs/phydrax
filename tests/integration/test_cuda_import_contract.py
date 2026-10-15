"""CUDA diagnostics and package imports must not load native providers."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(sys.platform != "linux", reason="Requires /proc/self/maps")
def test_package_imports_do_not_load_cuda_providers() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from pathlib import Path
import phydrax
import phydrax.backends
import phydrax.linalg
maps = Path('/proc/self/maps').read_text()
for name in ('libcudss', 'libamgxsh', 'libquda', 'libphydrax_cuda_'):
    assert name not in maps, name
""",
        ],
        cwd=_ROOT,
        env={**os.environ, "PYTHONPATH": str(_ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_static_cuda_collection_does_not_initialize_backends() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import phydrax
from phydrax._cuda_environment import collect_cuda_environment
facts = collect_cuda_environment(initialize_backend=False)
assert not facts.backends_initialized
assert not phydrax.config.snapshot().backends_initialized
""",
        ],
        cwd=_ROOT,
        env={**os.environ, "PYTHONPATH": str(_ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
