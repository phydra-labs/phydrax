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


@pytest.mark.parametrize("major", [None, 12, 13, 14])
def test_provider_import_admits_only_cpu_or_cuda13_installations(
    tmp_path: Path, major: int | None
) -> None:
    (tmp_path / "provider_fixture.py").write_text("loaded = True\n")
    if major is not None:
        directory = tmp_path / f"jax_cuda{major}_plugin-0.9.2.dist-info"
        directory.mkdir()
        (directory / "METADATA").write_text(
            f"Name: jax-cuda{major}-plugin\nVersion: 0.9.2\n"
        )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.metadata
import sys
from phydrax.backends._availability import probe_backend
from phydrax.backends._types import BackendCapabilities
directory, expected = sys.argv[1:]
distributions = tuple(importlib.metadata.distributions(path=[directory]))
importlib.metadata.distributions = lambda: distributions
sys.path.insert(0, directory)
capabilities = BackendCapabilities(
    backend="provider_fixture", problem_kinds=("linear.system",), execution="host",
    host_only=True, supports_matrix_free=False, supports_assembled=True,
    coordinate_dtypes=("float64",),
)
status = probe_backend(
    capabilities, module="provider_fixture", requirement="test provider",
    cuda_libraries=lambda: (),
)
accepted = expected == "accepted"
assert status.available == accepted, status.reason
assert ("provider_fixture" in sys.modules) == accepted
if not accepted:
    assert status.environment_issues[0].code == "plugin-major-mismatch", status.reason
    assert status.environment_issues[0].severity == "error"
""",
            str(tmp_path),
            "accepted" if major in (None, 13) else "refused",
        ],
        cwd=_ROOT,
        env={**os.environ, "PYTHONPATH": str(_ROOT), "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
