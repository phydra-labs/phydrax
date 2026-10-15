"""Rank-local GPU collectives, independent of JAX distributed bootstrap."""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_WORKER = r"""
import importlib.util
import json
import os
import sys


if any(importlib.util.find_spec(name) is None for name in ("mpi4jax", "mpi4py")):
    print("mpi4jax and mpi4py are required", flush=True)
    sys.exit(77)
from mpi4py import MPI
# This provider owns rank-local MPI, not JAX multi-host initialization. Retain
# the real MPI communicator while preventing the independent import bootstrap.
for key in ("SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE"):
    os.environ.pop(key, None)

import jax
import jax.numpy as jnp
import numpy as np
from phydrax._execution_device import device_platform
from phydrax.backends.distributed import Mpi4JaxCollectiveProvider, mpi4jax_availability

availability = mpi4jax_availability()
if availability.environment_issues:
    print(availability.reason, flush=True)
    sys.exit(77)

if not any(device_platform(device) == "cuda" for device in jax.devices()):
    print("A CUDA device is required", flush=True)
    sys.exit(77)

comm = MPI.COMM_WORLD.Dup()
direct = os.environ["MPI4JAX_USE_CUDA_MPI"] == "1"
try:
    provider = Mpi4JaxCollectiveProvider(comm, "cuda-mpi-test", gpu_direct=direct)
except RuntimeError as error:
    if str(error) not in (
        "mpi4jax was built without CUDA support",
        "MPI exposes no CUDA-awareness qualification query",
        "gpu_direct requires CUDA-aware MPI",
    ):
        raise
    print(str(error), flush=True)
    sys.exit(77)

if sys.argv[1] == "probe":
    print("CUDA-aware MPI qualified: " + MPI.Get_library_version(), flush=True)
    comm.Free()
    sys.exit(0)

rank = comm.Get_rank()
assert comm.Get_size() == 2
value = jnp.asarray([rank + 1, rank + 2], dtype=jnp.float32)
assert all(device_platform(device) == "cuda" for device in value.devices())
results = {}
for name, operation, reference in (
    ("sum", provider.sum, [3, 5]),
    ("mean", provider.mean, [1.5, 2.5]),
    ("maximum", provider.maximum, [2, 3]),
    ("minimum", provider.minimum, [1, 2]),
):
    result = jax.jit(operation)(value)
    result.block_until_ready()
    assert all(device_platform(device) == "cuda" for device in result.devices())
    np.testing.assert_array_equal(result, np.asarray(reference, dtype=np.float32))
    results[name] = np.asarray(result).tolist()

def exchange(value: jax.Array) -> jax.Array:
    return provider.sendrecv(value, source=1-rank, destination=1-rank)

received = jax.jit(exchange)(value)
received.block_until_ready()
np.testing.assert_array_equal(
    received, np.asarray([2-rank, 3-rank], dtype=np.float32)
)
results["sendrecv"] = np.asarray(received).tolist()
print(json.dumps({"rank": rank, "route": provider.provider_id, "values": results}), flush=True)
comm.Free()
"""


@pytest.mark.parametrize("gpu_direct", [True, False], ids=["cuda-direct", "host-staged"])
def test_cuda_mpi_collectives(gpu_direct: bool) -> None:
    launcher = shlex.split(os.environ.get("PHYDRAX_MPI_LAUNCHER", "mpirun"))
    if not launcher or shutil.which(launcher[0]) is None:
        pytest.skip("MPI launcher is not installed")
    environment = {
        **os.environ,
        "PYTHONPATH": str(_ROOT),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "MPI4JAX_USE_CUDA_MPI": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    probe = subprocess.run(
        [sys.executable, "-c", _WORKER, "probe"],
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
    environment["MPI4JAX_USE_CUDA_MPI"] = "1" if gpu_direct else "0"
    completed = subprocess.run(
        [*launcher, "-np", "2", sys.executable, "-c", _WORKER, "run"],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert '"rank": 0' in completed.stdout, completed.stdout
    assert '"rank": 1' in completed.stdout, completed.stdout


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
