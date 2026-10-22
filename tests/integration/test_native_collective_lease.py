"""Exercise native lease admission with real MPI and partitioned-domain fixtures."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def native_collective_lease(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("mpicxx")
    if compiler is None:
        pytest.skip(
            "Native collective lease tests require site Open MPI development files"
        )
    executable = tmp_path_factory.mktemp("native-collective-lease") / "lease-test"
    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            str(_ROOT / "tests" / "_support" / "collective_lease_native.cpp"),
            "-ldl",
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return executable


@pytest.mark.parametrize(
    "scenario", ["same-host", "partitioned", "split-error", "size-error", "free-error"]
)
def test_native_lease_shared_domain_admission(
    native_collective_lease: Path, scenario: str
) -> None:
    launcher = shlex.split(os.environ.get("PHYDRAX_MPI_LAUNCHER", "mpirun"))
    if not launcher or shutil.which(launcher[0]) is None:
        pytest.skip("Native collective lease tests require an MPI launcher")
    completed = subprocess.run(
        [*launcher, "-np", "2", str(native_collective_lease), scenario],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
