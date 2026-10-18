"""Qualify distinct-GPU and explicitly shared-GPU rank-local MPI collectives."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_WORKER = _ROOT / "tests" / "_support" / "cuda_mpi_worker.py"
_RANK_COUNT = int(os.environ.get("PHYDRAX_MPI_RANKS", "2"))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("gpu_direct", [True, False], ids=["cuda-direct", "host-staged"])
@pytest.mark.parametrize("placement", ["distinct-gpus", "shared-gpu"])
@pytest.mark.parametrize("rank_count", [_RANK_COUNT], ids=[f"ranks-{_RANK_COUNT}"])
def test_cuda_mpi_collectives(
    gpu_direct: bool, placement: str, dtype: str, rank_count: int
) -> None:
    assert rank_count >= 2, "Qualification requires at least two MPI ranks"
    launcher = shlex.split(os.environ.get("PHYDRAX_MPI_LAUNCHER", "mpirun"))
    if not launcher or shutil.which(launcher[0]) is None:
        pytest.skip(
            "Optional CUDA MPI qualification unavailable: MPI launcher is not installed"
        )
    environment = {
        **os.environ,
        "PYTHONPATH": str(_ROOT),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "MPI4JAX_USE_CUDA_MPI": "1" if gpu_direct else "0",
        "JAX_ENABLE_X64": "true",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    probe = subprocess.run(
        [sys.executable, str(_WORKER), "probe", placement, dtype, str(rank_count)],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if probe.returncode == 77:
        pytest.skip(probe.stdout.strip())
    assert probe.returncode == 0, probe.stdout + probe.stderr
    completed = subprocess.run(
        [
            *launcher,
            "-np",
            str(rank_count),
            sys.executable,
            str(_WORKER),
            "run",
            placement,
            dtype,
            str(rank_count),
        ],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    records = [
        json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")
    ]
    assert sorted(record["rank"] for record in records) == list(range(rank_count)), (
        completed.stdout
    )
    assert all(record["mpi_ranks"] == rank_count for record in records)
    assert all(record["shared_memory_ranks"] == rank_count for record in records)
    assert all(
        record["physical_multigpu"] is (placement == "distinct-gpus")
        for record in records
    )
    assert all(record["placement"] == placement for record in records)
    assert all(record["dtype"] == dtype for record in records)
    assert all(record["gpu_direct"] is gpu_direct for record in records)
    ownership = records[0]["device_ownership"]
    assert all(record["device_ownership"] == ownership for record in records)
    assert len(ownership) == rank_count
    assert len({uuid for _, uuid in ownership}) == (
        rank_count if placement == "distinct-gpus" else 1
    )
    assert len({host for host, _ in ownership}) == 1
    assert all(uuid.startswith("GPU-") for _, uuid in ownership)
    route = "cuda-direct" if gpu_direct else "host-staged"
    for record in records:
        phases = record["phases"]
        assert [phase["generation"] for phase in phases] == [0, 1]
        for generation, phase in enumerate(phases):
            assert phase["caller_released"] is True
            assert phase["route"] == f"mpi4jax:cuda-mpi-test-{generation}:{route}"
            assert [primal["step"] for primal in phase["primals"]] == list(
                range(generation * 3, generation * 3 + 3)
            )
            assert all(
                set(primal["values"]) == {"sum", "mean", "maximum", "minimum", "sendrecv"}
                for primal in phase["primals"]
            )
        assert set(phases[0]["differentiation"]) == {
            "sum.jvp",
            "mean.jvp",
            "sum.vjp",
            "mean.vjp",
            "sendrecv.vjp",
        }
        assert {
            name: refusal["exception"] for name, refusal in phases[0]["refusals"].items()
        } == {
            "maximum.jvp": "NotImplementedError",
            "minimum.jvp": "NotImplementedError",
            "maximum.vjp": "NotImplementedError",
            "minimum.vjp": "NotImplementedError",
            "sendrecv.jvp": "RuntimeError",
        }


def test_mismatched_cuda_mpi_bridge_is_refused_before_import() -> None:
    environment = {**os.environ, "PYTHONPATH": str(_ROOT)}
    environment.pop("JAX_PLATFORMS", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from phydrax._cuda_environment import (
    collect_cuda_environment, cuda_enabled, provider_dependency_closure,
    assess_provider_dependencies,
)
from phydrax.backends._availability import module_native_libraries
from phydrax.backends.distributed import mpi4jax_availability
facts = collect_cuda_environment(initialize_backend=False)
libraries = module_native_libraries(
    "mpi4jax", "_src/xla_bridge/mpi_xla_bridge_cuda*.so"
)
issues = tuple(
    issue
    for path in libraries
    for issue in assess_provider_dependencies(
        provider_dependency_closure(path, facts=facts, jax_backend_preloaded=False),
        facts,
    )
)
if not cuda_enabled(facts) or not any(
    issue.code == "provider-major-mismatch" for issue in issues
):
    print("The installed mpi4jax bridge does not mismatch the JAX CUDA plugin")
    sys.exit(77)
status = mpi4jax_availability()
assert not status.available
assert any(issue.code == "provider-major-mismatch" for issue in status.environment_issues), status.reason
assert all(issue.message in status.reason for issue in status.environment_issues)
assert "mpi4jax" not in sys.modules
print(status.reason)
""",
        ],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode == 77:
        pytest.skip(completed.stdout.strip())
    assert completed.returncode == 0, completed.stdout + completed.stderr
