"""Companion teardown drains in-flight GPU work before native finalization."""

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from textwrap import indent

import pytest

from phydrax._cuda_native import CudaNativeComponentName


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
matrix = jnp.asarray([[4., 1.], [1., 3.]], dtype=jnp.float64) * (DEVICE_INDEX + 1)
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
ready = la.solve(prepared, rhs)
ready.value.block_until_ready()
assert bool(ready.successful)
assert ready.value.devices() == {device}
np.testing.assert_allclose(ready.value, np.linalg.solve(np.asarray(matrix), np.asarray(rhs)), rtol=1e-10)
from phydrax._cuda_native import cuda_native_resources
assert any(p.component == "cudss" and p.live > 0 for p in cuda_native_resources().partitions)
result = la.solve(prepared, rhs)
""",
    "amgx": _PROBLEM
    + """
from phydrax.backends import AmgXPolicy, plan_amgx, prepare_amgx, solve_amgx
plan = plan_amgx(system, AmgXPolicy())
prepared = prepare_amgx(system, plan)
ready = solve_amgx(prepared, rhs)
ready.value.block_until_ready()
assert bool(ready.success), {
    "device": str(device),
    "device_index": DEVICE_INDEX,
    "value": np.asarray(ready.value).tolist(),
    "status": np.asarray(ready.status).tolist(),
    "native_status": np.asarray(ready.diagnostics.native_status).tolist(),
    "iterations": np.asarray(ready.diagnostics.iterations).tolist(),
    "relative_residual": np.asarray(ready.diagnostics.relative_residual_norm).tolist(),
    "native_residual": np.asarray(ready.diagnostics.native_residual_norm).tolist(),
}
assert ready.value.devices() == {device}
np.testing.assert_allclose(ready.value, np.linalg.solve(np.asarray(matrix), np.asarray(rhs)), rtol=1e-8)
from phydrax._cuda_native import cuda_native_resources
assert any(p.component == "amgx" and p.live > 0 for p in cuda_native_resources().partitions)
result = solve_amgx(prepared, rhs)
""",
    "quda": """
from phydrax.backends import create_lattice_provider
provider = create_lattice_provider("quda")
links = jnp.broadcast_to(jnp.eye(3, dtype=jnp.complex64), (256, 4, 3, 3))
mask = jnp.ones((256, 6), dtype=jnp.bool_)
ready = provider.gauge_action(links, 5.7, mask, lattice_shape=(4, 4, 4, 4), periodic=(True,) * 4)
ready.block_until_ready()
assert ready.devices() == {device}
np.testing.assert_allclose(ready, 0., atol=1e-5)
result = provider.gauge_action(links, 5.7, mask, lattice_shape=(4, 4, 4, 4), periodic=(True,) * 4)
""",
}
_AUDIT = """
from phydrax._cuda_environment import cuda_environment
from phydrax._cuda_environment import competing_cuda_libraries
report = cuda_environment()
assert report.facts.runtime_version[0] == component.major == 13
assert component.preloaded, component
wheel_paths = {path for path, major in report.facts.wheel_library_majors if major == component.major}
assert all(path in wheel_paths for path in component.preloaded), component.preloaded
assert all(item.major == component.major and item.path in wheel_paths for item in report.facts.mapped_libraries), report
challenge = competing_cuda_libraries(HOSTILE_LIBRARY_PATH, report.facts.mapped_libraries)
assert challenge, "Hostile library path contains no alternate mapped CUDA SONAME"
audit = {"preloaded": component.preloaded, "mapped": [(item.path, item.major) for item in report.facts.mapped_libraries], "hostile_candidates": challenge}
"""
# A system CUDA toolkit directory whose sonames can shadow NVIDIA wheel libraries.
_HOSTILE_LIBRARY_PATH = os.environ.get(
    "PHYDRAX_CUDA_HOSTILE_LIBRARY_PATH", "/usr/local/cuda/lib64"
)


def _run(
    component: str, *, hostile: bool, device_indices: list[int]
) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    environment.pop("LD_LIBRARY_PATH", None)
    if hostile:
        environment["LD_LIBRARY_PATH"] = _HOSTILE_LIBRARY_PATH
    source = (
        f"COMPONENT = {component!r}\nHOSTILE = {hostile!r}\n"
        f"HOSTILE_LIBRARY_PATH = {_HOSTILE_LIBRARY_PATH!r}\n" + _SETUP
    )
    operations = (
        "from phydrax._jax_config import default_device\n"
        "devices = jax.devices('gpu')\n"
        f"indices = {device_indices!r}\n"
        "assert indices and max(indices) < len(devices), 'Required native exit devices unavailable'\n"
        "retained = []\n"
        "covered = []\n"
        "for DEVICE_INDEX in indices:\n"
        "    device = devices[DEVICE_INDEX]\n"
        "    with default_device(device):\n"
        + indent(_OPERATIONS[component], "        ")
        + "\n        retained.append(result if COMPONENT == 'quda' else (prepared, result))\n"
        "        covered.append(device.id)\n"
    )
    if component != "quda":
        operations += (
            "partitions = [p for p in cuda_native_resources().partitions "
            "if p.component == COMPONENT and p.live > 0]\n"
            "assert {p.device for p in partitions} == set(covered)\n"
            "assert len({p.partition for p in partitions}) == 1\n"
        )
        operations = (
            operations + "\nfrom phydrax._cuda_native import _shutdown\n"
            "_shutdown()\n"
            "assert not any(p.live for p in cuda_native_resources().partitions)\n"
            + operations
        )
    operations += _AUDIT if hostile else "\naudit = {}\n"
    operations += '\nprint(json.dumps({"devices": covered, **audit}), flush=True)\n'
    source += "\nwith config.context(config.NumericsConfig(enable_x64=True)):\n"
    source += indent(operations, "    ")
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=(
            Path.cwd()
            if "PHYDRAX_CUDA_QUALIFICATION_GPUS" in environment
            else Path(__file__).resolve().parents[2]
        ),
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
    )


@pytest.mark.cuda_native
@pytest.mark.parametrize("component", ["cudss", "amgx", "quda"])
@pytest.mark.parametrize("hostile", [False, True], ids=["normal", "hostile-library-path"])
def test_exit_with_live_native_state_and_inflight_work(
    component: CudaNativeComponentName,
    hostile: bool,
    record_property: Callable[[str, object], None],
) -> None:
    import jax

    from phydrax._cuda_environment import competing_cuda_libraries, cuda_environment
    from phydrax._cuda_native import CudaNativeUnavailableError, load_cuda_component

    try:
        load_cuda_component(component)
    except CudaNativeUnavailableError as error:
        if error.issue.code == "backend-unavailable" or (
            error.issue.code == "companion-missing" and not error.issue.paths
        ):
            pytest.skip(str(error))
        raise
    if hostile:
        report = cuda_environment()
        if not competing_cuda_libraries(
            _HOSTILE_LIBRARY_PATH, report.facts.mapped_libraries
        ):
            pytest.skip(
                "No alternate mapped CUDA SONAME exists in the hostile library path."
            )
    visible = jax.devices("gpu")
    count = int(os.environ.get("PHYDRAX_CUDA_QUALIFICATION_GPUS", len(visible)))
    assert 0 < count <= len(visible), "Required native exit devices are unavailable"
    groups = (
        [[index] for index in range(count)]
        if component == "quda"
        else [list(range(count))]
    )
    covered = []
    for indices in groups:
        result = _run(component, hostile=hostile, device_indices=indices)
        if result.returncode == 77:
            pytest.skip(result.stdout.strip())
        if result.returncode == 78:
            refusal = json.loads(result.stdout.splitlines()[-1])
            if refusal["refused"] == "companion-missing":
                pytest.skip(refusal["message"])
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
        evidence = json.loads(result.stdout.splitlines()[-1])
        assert evidence["devices"] == [visible[index].id for index in indices]
        covered.extend(evidence["devices"])
        if hostile:
            assert evidence["preloaded"]
            assert evidence["mapped"]
    record_property("cuda_native_devices", json.dumps(covered))
