# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Compiler and topology admission behavior, independent of hardware qualification."""

import gc
import sys
import weakref
from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from types import ModuleType, SimpleNamespace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.lib import xla_client

import phydrax as phx
import phydrax._cuda_capabilities as capabilities
from phydrax.backends._cuda_profiles import cuda_candidate_profiles
from phydrax.backends._types import BackendUnavailableError
from phydrax.backends.cudecomp import cudecomp_availability
from phydrax.backends.cudss import cudss_availability, CudssExecutionMode
from phydrax.backends.distributed import mpi4jax_availability, NVIDIA_JAX_PROFILE


class _CudaClient:
    platform = "cuda"
    platform_version = "cuda 13000"

    def devices(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(platform="gpu", client=self, compute_capability="12.0")]


def _cuda_client() -> xla_client.Client:
    return cast(xla_client.Client, _CudaClient())


def _compiler() -> capabilities.CudaCompilerCapabilities:
    return capabilities.CudaCompilerCapabilities(
        host_platform="win32",
        distribution="phydrax-cuda13-pjrt",
        version="test",
        build_id="build-a",
        jax_version=jax.__version__,
        jaxlib_version=capabilities.version("jaxlib"),
        platform_version="cuda 13000",
    )


def test_cpu_destination_does_not_inspect_cuda_or_global_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = jax.devices("cpu")[0].client

    def forbidden(*args: object) -> None:
        raise AssertionError("CPU compiler admission must not inspect global CUDA intent")

    monkeypatch.setattr(jax, "default_backend", forbidden)
    monkeypatch.setattr(capabilities, "distribution", forbidden)
    compiler = capabilities.cuda_compiler_capabilities(client)
    assert not compiler.mosaic_gpu
    assert not compiler.pallas_triton
    assert not compiler.sparse_qr
    assert compiler.platform_version == client.platform_version


@pytest.mark.parametrize(
    "declared", [(), ("sparse_qr",), ("mosaic_gpu", "pallas_triton", "sparse_qr")]
)
@pytest.mark.parametrize("native", [False, True])
def test_windows_requires_manifest_and_native_kernel_support(
    monkeypatch: pytest.MonkeyPatch,
    declared: tuple[str, ...],
    native: bool,
) -> None:
    monkeypatch.setattr(capabilities, "sys", SimpleNamespace(platform="win32"))
    record = SimpleNamespace(
        distribution="phydrax-cuda13-pjrt",
        version="test",
        build_id="actual-build",
        jax_version=jax.__version__,
        jaxlib_version=capabilities.version("jaxlib"),
        capabilities=declared,
        issues=(),
    )
    runtime = ModuleType("phydrax_cuda13_pjrt.runtime")
    setattr(runtime, "installation_record", lambda: record)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setattr(
        capabilities, "_native_support", lambda _: (native, native, native)
    )
    compiler = capabilities.cuda_compiler_capabilities(_cuda_client())
    assert compiler.mosaic_gpu == (native and "mosaic_gpu" in declared)
    assert compiler.pallas_triton == (native and "pallas_triton" in declared)
    assert compiler.sparse_qr == (native and "sparse_qr" in declared)
    assert compiler.build_id == record.build_id
    assert compiler.identity != replace(compiler, build_id="other-build").identity
    assert compiler.identity != replace(compiler, host_platform="linux").identity
    assert compiler.identity_payload["platform_version"] == "cuda 13000"


def test_invalid_windows_manifest_retains_inspection_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capabilities, "sys", SimpleNamespace(platform="win32"))
    runtime = ModuleType("phydrax_cuda13_pjrt.runtime")

    def invalid() -> None:
        raise ValueError("runtime manifest digest mismatch")

    setattr(runtime, "installation_record", invalid)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    compiler = capabilities.cuda_compiler_capabilities(_cuda_client())
    assert (
        not compiler.mosaic_gpu and not compiler.pallas_triton and not compiler.sparse_qr
    )
    assert "runtime manifest digest mismatch" in compiler.reason


def test_linux_matched_plugin_support_is_not_narrowed_to_windows_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capabilities, "sys", SimpleNamespace(platform="linux"))

    def installed(name: str) -> SimpleNamespace:
        if not name.startswith("jax-cuda13-"):
            raise PackageNotFoundError(name)
        return SimpleNamespace(
            version=capabilities.version("jaxlib"),
            read_text=lambda _: f"{name}/native.so,sha256=actual-record-hash,123",
        )

    monkeypatch.setattr(capabilities, "distribution", installed)
    monkeypatch.setattr(capabilities, "_native_support", lambda _: (True, True, True))
    compiler = capabilities.cuda_compiler_capabilities(_cuda_client())
    assert compiler.mosaic_gpu and compiler.pallas_triton and compiler.sparse_qr
    assert compiler.distribution == "jax-cuda13-pjrt"
    assert compiler.build_id
    assert (
        capabilities.windows_topology_refusal(
            device_count=8, process_count=8, site_mpi=True
        )
        is None
    )


@pytest.mark.parametrize("foreign", ["jax-cuda12-plugin", "jax-cuda12-pjrt"])
@pytest.mark.parametrize("cuda13_installed", [False, True])
def test_linux_foreign_cuda_major_refuses_even_with_native_entry_points(
    monkeypatch: pytest.MonkeyPatch,
    foreign: str,
    cuda13_installed: bool,
) -> None:
    monkeypatch.setattr(capabilities, "sys", SimpleNamespace(platform="linux"))

    def installed(name: str) -> SimpleNamespace:
        if name != foreign and not (cuda13_installed and name.startswith("jax-cuda13-")):
            raise PackageNotFoundError(name)
        return SimpleNamespace(
            version=capabilities.version("jaxlib"),
            read_text=lambda _: f"{name}/native.so,sha256=actual-record-hash,123",
        )

    monkeypatch.setattr(capabilities, "distribution", installed)
    monkeypatch.setattr(capabilities, "_native_support", lambda _: (True, True, True))
    compiler = capabilities.cuda_compiler_capabilities(_cuda_client())
    assert (
        not compiler.mosaic_gpu and not compiler.pallas_triton and not compiler.sparse_qr
    )
    assert compiler.distribution == foreign
    assert "CUDA 12 is unsupported" in compiler.reason


@pytest.mark.parametrize("target", ["cusolver_csrlsvqr_ffi", "unrelated_kernel"])
def test_sparse_qr_requires_its_registered_solver_kernel(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    from jaxlib import gpu_solver, gpu_triton

    solver = SimpleNamespace(
        __name__="jax_cuda13_plugin._solver",
        registrations=lambda: {target: object()},
    )
    monkeypatch.setattr(gpu_solver, "_cusolver", solver)
    monkeypatch.setattr(gpu_triton, "_cuda_triton", None)
    monkeypatch.setattr(capabilities, "_optional_module", lambda _: None)
    mosaic, triton, sparse = capabilities._native_support("jax_cuda13_plugin")
    assert not mosaic and not triton
    assert sparse == (target == "cusolver_csrlsvqr_ffi")
    assert capabilities._native_support("jax_cuda12_plugin") == (False, False, False)


def test_sparse_qr_missing_kernel_refuses_explicit_request_and_avoids_autoselection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.linalg._sparse_providers as providers

    compiler = _compiler()
    monkeypatch.setattr(providers, "cuda_compiler_capabilities", lambda _: compiler)
    la = phx.linalg
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1, 0, 1], dtype=jnp.int32),
        jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
        source_size=2,
        target_size=2,
    )
    operator = phx.sparse.SparseLinearMap(
        relation,
        jnp.asarray([4.0, 1.0, 1.0, 3.0]),
        operator_id="compiler-admission",
        properties=la.OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "asserted"}
        ),
    )
    problem = la.LinearSystem(operator)
    policy = la.LinearSolvePolicy(failure=la.FailurePolicy("error"))
    selected = la.plan(problem, policy)
    assert selected.method != "sparse-qr"
    assert any("sparse QR kernel" in reason for reason in selected.rejected)
    rhs = jnp.asarray([1.0, -2.0])
    result = la.solve(problem, rhs, policy=policy)
    np.testing.assert_allclose(
        result.value, np.linalg.solve([[4.0, 1.0], [1.0, 3.0]], [1.0, -2.0])
    )
    with pytest.raises(ValueError, match="sparse QR kernel"):
        la.plan(problem, la.LinearSolvePolicy(la.SparseQR(provider="jax-cuda")))
    compiler = replace(compiler, sparse_qr=True)
    assert la.plan(problem, policy).method == "sparse-qr"
    assert (
        la.plan(problem, la.LinearSolvePolicy(la.SparseQR(provider="jax-cuda"))).method
        == "sparse-qr"
    )


@pytest.mark.parametrize("mode", ["mg", "mgmn"])
def test_windows_topology_refusals_preserve_provider_exception(
    monkeypatch: pytest.MonkeyPatch,
    mode: CudssExecutionMode,
) -> None:
    monkeypatch.setattr(capabilities, "sys", SimpleNamespace(platform="win32"))
    assert capabilities.windows_topology_refusal() is None
    for availability, operation in (
        (cudss_availability(execution_mode=mode), "linear.sparse-system"),
        (cudecomp_availability(), "spectral.direct-slab-transpose"),
        (mpi4jax_availability(), "collective.sum"),
    ):
        assert not availability.available
        with pytest.raises(
            BackendUnavailableError, match="one process and one device"
        ) as caught:
            availability.require(operation)
        assert caught.value.backend == availability.backend
        assert caught.value.capability == operation
    assert "multi_device" in NVIDIA_JAX_PROFILE.scopes
    assert "multi_host" in NVIDIA_JAX_PROFILE.scopes


def test_cuda_candidates_separate_windows_local_support_from_linux_topologies() -> None:
    windows = []
    linux = []
    for profile in cuda_candidate_profiles():
        for support in profile.support_tuples:
            attributes = dict(support.attributes)
            (windows if attributes["host_platform"] == "win32" else linux).append(
                attributes
            )
    assert windows and linux
    assert {entry["provider"] for entry in windows} == {"cudss", "cutensor"}
    assert all(entry["device_count"] == entry["process_count"] == 1 for entry in windows)
    assert all(entry["runtime"] == "phydrax-cuda13-pjrt" for entry in windows)
    assert any(entry["provider"] == "cudecomp" for entry in linux)
    assert any(entry.get("execution") == "mgmn" for entry in linux)


def test_compiler_snapshot_does_not_keep_destination_client_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capabilities, "sys", SimpleNamespace(platform="unknown"))
    client = _cuda_client()
    reference = weakref.ref(client)
    compiler = capabilities.cuda_compiler_capabilities(client)
    assert capabilities.cuda_compiler_capabilities(client) is compiler
    del client
    gc.collect()
    assert reference() is None
    assert compiler.platform_version == "cuda 13000"


def test_windows_cudss_mg_plan_refuses_before_device_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._execution_resources import ExecutionGroupSpec

    monkeypatch.setattr(capabilities, "sys", SimpleNamespace(platform="win32"))
    la = phx.linalg
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray([0, 1], dtype=jnp.int32),
        source_size=2,
        target_size=2,
    )
    operator = phx.sparse.SparseLinearMap(
        relation,
        jnp.asarray([2.0, 3.0]),
        operator_id="windows-mg-admission",
        properties=la.OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "asserted"}
        ),
    )
    group = ExecutionGroupSpec("windows-mg", (0,), ((0, 0), (0, 1)))
    method = la.SparseLDLT(execution=la.CudssExecutionPolicy("mg", group=group))
    with pytest.raises(ValueError, match="one process and one device"):
        la.plan(la.LinearSystem(operator), la.LinearSolvePolicy(method))
