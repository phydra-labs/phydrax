"""Companion teardown drains in-flight GPU work before native finalization."""

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import indent

import pytest

from phydrax._cuda_environment import collect_cuda_environment


_SETUP = """
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from phydrax._cuda_native import load_cuda_component, CudaNativeUnavailableError
from phydrax import config
try:
    component = load_cuda_component(COMPONENT)
except CudaNativeUnavailableError as error:
    print(json.dumps({"refused": error.issue.code, "paths": list(error.issue.paths), "message": str(error)}))
    raise SystemExit(78 if HOSTILE else 77)
"""
_PROBLEM = """
import phydrax as phx
la = phx.linalg
matrix = jnp.asarray([[4., 1.], [1., 3.]], dtype=jnp.float64)
rows, columns = np.indices((2, 2), dtype=np.int32)
relation = phx.sparse.EdgeRelation(columns.reshape(-1), rows.reshape(-1), source_size=2, target_size=2)
operator = phx.sparse.SparseLinearMap(relation, matrix.reshape(-1), operator_id="native-exit", properties=la.OperatorProperties(self_adjoint=True, evidence={"self_adjoint": "asserted"}))
system = la.LinearSystem(operator)
rhs = jnp.asarray([1., -2.], dtype=jnp.float64)
"""
_OPERATIONS = {
    "cudss": _PROBLEM
    + """
prepared = la.prepare(system, la.LinearSolvePolicy(la.SparseLDLT(provider="cudss")))
# Materialize the memo but deliberately retain it and do not block the final solve.
la.solve(prepared, rhs).value.block_until_ready()
from phydrax._cuda_native import cuda_native_resources
assert any(p.component == "cudss" and p.live > 0 for p in cuda_native_resources().partitions)
result = la.solve(prepared, rhs)
""",
    "amgx": _PROBLEM
    + """
from phydrax.backends import AmgXPolicy, plan_amgx, prepare_amgx, solve_amgx
plan = plan_amgx(system, AmgXPolicy())
prepared = prepare_amgx(system, plan)
solve_amgx(prepared, rhs).value.block_until_ready()
from phydrax._cuda_native import cuda_native_resources
assert any(p.component == "amgx" and p.live > 0 for p in cuda_native_resources().partitions)
result = solve_amgx(prepared, rhs)
""",
    "quda": """
from phydrax.backends import create_lattice_provider
provider = create_lattice_provider("quda")
links = jnp.broadcast_to(jnp.eye(3, dtype=jnp.complex64), (256, 4, 3, 3))
mask = jnp.ones((256, 6), dtype=jnp.bool_)
provider.gauge_action(links, 5.7, mask, lattice_shape=(4, 4, 4, 4), periodic=(True,) * 4).block_until_ready()
result = provider.gauge_action(links, 5.7, mask, lattice_shape=(4, 4, 4, 4), periodic=(True,) * 4)
""",
}
_AUDIT = """
from phydrax._cuda_environment import cuda_environment
report = cuda_environment()
assert report.facts.runtime_version[0] == component.major
assert component.preloaded, component
wheel_paths = {path for path, major in report.facts.wheel_library_majors if major == component.major}
assert all(path in wheel_paths for path in component.preloaded), component.preloaded
assert all(item.major == component.major and item.path in wheel_paths for item in report.facts.mapped_libraries), report
print(json.dumps({"preloaded": component.preloaded, "mapped": [(item.path, item.major) for item in report.facts.mapped_libraries]}))
"""
# A system CUDA 12 toolkit directory: its sonames shadow NVIDIA wheel libraries.
_HOSTILE_LIBRARY_PATH = "/usr/local/cuda/lib64"


def _run(component: str, *, hostile: bool) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    if hostile:
        environment["LD_LIBRARY_PATH"] = _HOSTILE_LIBRARY_PATH
    source = f"COMPONENT = {component!r}\nHOSTILE = {hostile!r}\n" + _SETUP
    operations = _OPERATIONS[component]
    if hostile:
        operations += _AUDIT
    source += "\nwith config.context(config.NumericsConfig(enable_x64=True)):\n"
    source += indent(operations, "    ")
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
    )


@pytest.mark.cuda_native
@pytest.mark.parametrize("component", ["cudss", "amgx", "quda"])
@pytest.mark.parametrize("hostile", [False, True], ids=["normal", "hostile-library-path"])
def test_exit_with_live_native_state_and_inflight_work(
    component: str, hostile: bool
) -> None:
    result = _run(component, hostile=hostile)
    if result.returncode == 77:
        pytest.skip(result.stdout.strip())
    if result.returncode == 78:
        refusal = json.loads(result.stdout.splitlines()[-1])
        if refusal["refused"] == "companion-missing":
            pytest.skip(refusal["message"])
        # CUDA 13 must survive the shadowing path; JAX's CUDA 12 plugin may not
        # initialize under it, and the refusal must then name the directory.
        plugins = collect_cuda_environment(initialize_backend=False).plugin_majors
        assert plugins == (12,), refusal
        assert refusal["refused"] == "backend-unavailable", refusal
        assert _HOSTILE_LIBRARY_PATH in refusal["paths"], refusal
        return
    assert result.returncode == 0, result.stdout + result.stderr
    output = (result.stdout + result.stderr).lower()
    for error in (
        "cuda error",
        "cudaerror",
        "exception ignored in atexit",
        "shutdown_failed",
        "invalid device context",
    ):
        assert error not in output, result.stdout + result.stderr
    if hostile:
        evidence = json.loads(result.stdout.splitlines()[-1])
        assert evidence["preloaded"]
        assert evidence["mapped"]
