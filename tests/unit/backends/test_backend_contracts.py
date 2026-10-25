#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import os
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

import jax.numpy as jnp
import pytest

import phydrax as phx


@pytest.mark.parametrize(
    "probe",
    [
        pytest.param(phx.backends.quda_availability, marks=pytest.mark.cuda_native),
        phx.backends.grid_availability,
        phx.backends.qlat_availability,
    ],
    ids=("quda", "grid", "qlat"),
)
def test_external_lattice_availability_admits_only_executable_adapters(
    probe: Any,
) -> None:
    from phydrax.backends.lattice import create_lattice_provider

    status = probe()
    if not status.available:
        with pytest.raises(phx.backends.BackendUnavailableError):
            status.require(status.capabilities.problem_kinds[0])
        return
    provider = create_lattice_provider(status.backend)
    assert provider.status().available
    assert provider.capabilities.operations == status.capabilities.problem_kinds
    assert provider.capabilities.dtypes == status.capabilities.coordinate_dtypes
    assert (
        provider.capabilities.transformations
        == status.capabilities.distribution.transformations
    )


def _capabilities() -> Any:
    return phx.backends.BackendCapabilities(
        backend="contract-probe",
        problem_kinds=("linear.system",),
        execution="host",
        host_only=True,
        supports_matrix_free=False,
        supports_assembled=True,
        coordinate_dtypes=("float64",),
    )


def test_backend_probe_is_lazy_and_reports_missing_requirement_exactly(
    phydrax_events: Any,
) -> None:
    availability = phx.backends.probe_backend(
        _capabilities(),
        module="__phydrax_backend_that_does_not_exist__",
        requirement="missing-test-provider>=1",
        distributions=("__phydrax_missing_distribution__",),
    )

    assert not availability.available
    assert availability.backend == "contract-probe"
    assert availability.versions == ()
    assert availability.environment_issues == ()
    assert "not installed" in availability.reason
    with pytest.raises(phx.backends.BackendUnavailableError) as error:
        availability.require("linear.system")
    assert "contract-probe" in str(error.value)
    assert "missing-test-provider>=1" in str(error.value)
    assert "linear.system" in str(error.value)
    event = phydrax_events.records("backend.probe.completed")[-1]
    assert event["fields"]["backend"] == "contract-probe"
    assert event["fields"]["failure_category"] == "not_installed"
    assert "__phydrax_backend_that_does_not_exist__" not in (
        phydrax_events.path.read_text(encoding="utf-8")
    )


def test_backend_contracts() -> None:
    availability = phx.backends.BackendAvailability(
        capabilities=_capabilities(),
        available=True,
        requirement="installed-provider",
        reason="provider module imported successfully",
        versions=(("installed-provider", "1.2.3"),),
    )

    availability.require("linear.system")
    with pytest.raises(phx.backends.BackendUnavailableError, match="does not declare"):
        availability.require("eigen.general")
    evidence = phx.backends.BackendTransferEvidence(
        host_to_device_bytes=128,
        device_to_host_bytes=64,
        synchronization_count=2,
    )

    assert evidence.host_to_device_bytes.dtype == jnp.int64
    assert int(evidence.host_to_device_bytes) == 128
    assert int(evidence.device_to_host_bytes) == 64
    assert int(evidence.synchronization_count) == 2


def _cuda_subprocess(source: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[3]
    environment = {**os.environ, "PYTHONPATH": str(root)}
    environment.pop("JAX_PLATFORMS", None)
    completed = subprocess.run(
        [sys.executable, "-c", source, *arguments],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode == 77:
        pytest.skip(completed.stdout.strip())
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def test_cuda_provider_probe_order_is_independent() -> None:
    source = """
from phydrax._cuda_environment import collect_cuda_environment
import sys
if tuple(plugin.cuda_major for plugin in collect_cuda_environment(initialize_backend=False).plugins) != (13,):
    print("The JAX CUDA13 plugin distribution is required")
    sys.exit(77)
from phydrax.backends.cudss import cudss_availability
from phydrax.backends.amg import amgx_availability
from phydrax.backends.lattice import quda_availability
if sys.argv[1] == "providers-first":
    amgx_availability()
    quda_availability()
status = cudss_availability()
import json
print("CUDSS-STATUS " + json.dumps([status.available, status.reason]), flush=True)
"""

    def cudss_status(completed: subprocess.CompletedProcess[str]) -> str:
        # Native providers (QUDA) print their own banners and exit reports.
        lines = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("CUDSS-STATUS ")
        ]
        assert len(lines) == 1, completed.stdout
        return lines[0]

    first = _cuda_subprocess(source, "cudss-first")
    second = _cuda_subprocess(source, "providers-first")
    assert cudss_status(first) == cudss_status(second)


def test_cuda_provider_dependency_mismatch_refuses_before_import(tmp_path: Path) -> None:
    _cuda_subprocess(
        """
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from phydrax._cuda_environment import collect_cuda_environment
facts = collect_cuda_environment(initialize_backend=False)
if tuple(plugin.cuda_major for plugin in facts.plugins) != (13,):
    print("The JAX CUDA13 plugin distribution is required")
    sys.exit(77)
from tests._support.elf import write_shared_object
directory = Path(sys.argv[1])
soname = "libcublas.so.12"
write_shared_object(directory / soname, soname=soname)
extension = write_shared_object(
    directory / ("fakeprov" + EXTENSION_SUFFIXES[0]),
    needed=(soname,), runpath=("$ORIGIN",),
)
sys.path.insert(0, str(directory))
from phydrax.backends._availability import probe_backend
from phydrax.backends._types import BackendCapabilities
capabilities = BackendCapabilities(
    backend="fakeprov", problem_kinds=("linear.system",), execution="host",
    host_only=True, supports_matrix_free=False, supports_assembled=True,
    coordinate_dtypes=("float64",),
)
status = probe_backend(
    capabilities, module="fakeprov", requirement="test provider",
    cuda_libraries=lambda: (str(extension),),
)
assert not status.available, status.reason
assert "fakeprov" not in sys.modules
assert status.environment_issues[0].code == "provider-major-mismatch", status.reason
""",
        str(tmp_path),
    )


@pytest.mark.parametrize("exception", [RuntimeError, ImportError])
def test_windows_native_inspection_failure_is_a_probe_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    phydrax_events: Any, exception: type[Exception],
) -> None:
    from types import ModuleType

    from phydrax._cuda_environment import CudaEnvironmentFacts, CudaPluginInstallation
    from phydrax.backends import _availability as availability
    from tests._support.pe import write_pe

    path = write_pe(tmp_path / "provider.dll")
    module = "inspection_refusal_provider"
    (tmp_path / f"{module}.py").write_text(
        'raise AssertionError("refused provider must not import")', encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    facts = CudaEnvironmentFacts(
        host_platform="win32", host_machine="AMD64",
        plugins=(CudaPluginInstallation(
            distribution="phydrax-cuda13-pjrt", version="0.2.2", cuda_major=13,
        ),),
    )
    monkeypatch.setattr(availability, "collect_cuda_environment", lambda **kwargs: facts)
    runtime = ModuleType("phydrax_cuda13_pjrt.runtime")

    def refused(path: str) -> NoReturn:
        raise exception("Unowned resident CUDA DLL family member foreign/cudnn.dll")

    setattr(runtime, "dependency_closure", refused)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    result = availability.probe_backend(
        _capabilities(), module=module, requirement="test provider",
        cuda_libraries=lambda: (str(path),),
    )
    assert not result.available
    assert "Unowned resident CUDA DLL family member" in result.reason
    assert module not in sys.modules
    event = phydrax_events.records("backend.probe.completed")[-1]
    assert event["fields"]["failure_category"] == "cuda_environment_refused"
