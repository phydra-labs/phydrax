"""Real C ABI fixtures exercise companion refusals before FFI registration."""

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from phydrax import _cuda_native as native
from phydrax._cuda_environment import CudaEnvironmentFacts


@pytest.fixture
def isolated_loader(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    registrations: list[str] = []
    facts = CudaEnvironmentFacts(
        host_platform="linux", host_machine="x86_64", plugin_majors=(13,)
    )
    monkeypatch.setattr(native, "_cuda_facts", lambda: facts)
    monkeypatch.setattr(native, "_LOADED", {})
    monkeypatch.setattr(native, "_EXIT_REGISTERED", True)

    def register(name: str, capsule: object, *, platform: str) -> None:
        assert platform == "CUDA"
        registrations.append(name)

    monkeypatch.setattr(jax.ffi, "register_ffi_target", register)
    return registrations


def _compile_library(
    tmp_path: Path,
    *,
    component: str = "cudss",
    version: str | None = None,
    major: int = 13,
    missing_handler: bool = False,
    dependency: bool = False,
    omit_symbol: bool = False,
    omit_handler_symbol: bool = False,
) -> Path:
    compiler = shutil.which("gcc")
    if compiler is None:
        pytest.skip("The C ABI refusal fixtures require gcc.")
    names = (
        [
            f"phydrax_cudss_{op}_{dtype}"
            for op in ("analyze", "factorize", "refactorize", "solve", "diagonal")
            for dtype in ("f32", "f64")
        ]
        + ["phydrax_cudss_release"]
        if component == "cudss"
        else [
            "phydrax_amgx_setup",
            "phydrax_amgx_refresh",
            "phydrax_amgx_solve",
            "phydrax_amgx_release",
        ]
    )
    if missing_handler:
        names.pop()
    source = tmp_path / f"{component}.c"
    library = tmp_path / f"libfixture_{component}.so"
    release_version = (
        importlib.metadata.version("phydrax") if version is None else version
    )
    declarations = ",".join(json.dumps(name) for name in names)
    exports = "\n".join(
        f"void {name}(void) {{}}"
        for name in names
        if not (omit_handler_symbol and name == names[-1])
    )
    source.write_text(f'''
#include <stdint.h>
#include <stddef.h>
static const char *names[] = {{{declarations}}};
static void handler(void) {{}}
{exports}
const char *phx_cuda_version(void) {{ return "{release_version}"; }}
const char *phx_cuda_build_hash(void) {{ return "{"a" * 64}"; }}
const char *phx_cuda_component(void) {{ return "{component}"; }}
const char *phx_cuda_last_error(void) {{ return ""; }}
int32_t phx_cuda_major(void) {{ return {major}; }}
int32_t phx_cuda_handler_count(void) {{ return sizeof(names)/sizeof(names[0]); }}
const char *phx_cuda_handler_name(int32_t i) {{ return names[i]; }}
void *phx_cuda_handler(int32_t i) {{ return (void *)handler; }}
int32_t phx_cuda_registry_partitions(void) {{ return 0; }}
int32_t phx_cuda_registry_partition(int32_t i, int32_t *d, int64_t *p, int64_t *l, int64_t *c, int64_t *r) {{ return 1; }}
int32_t phx_cuda_release(int64_t p, const int32_t *ids, int64_t count, int64_t *released) {{ *released=0; return 0; }}
int32_t phx_cuda_release_all(int64_t *released) {{ *released=0; return 0; }}
{"" if omit_symbol else "int32_t phx_cuda_shutdown(void) { return 0; }"}
''')
    command = [compiler, "-shared", "-fPIC", "-o", str(library), str(source)]
    if dependency:
        dependency_source = tmp_path / "dependency.c"
        dependency_source.write_text("void fixture_dependency(void) {}\n")
        dependency_library = tmp_path / "libcudss.so.0"
        subprocess.run(
            [
                compiler,
                "-shared",
                "-fPIC",
                "-Wl,-soname,libcudss.so.0",
                "-o",
                str(dependency_library),
                str(dependency_source),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        command += ["-Wl,--no-as-needed", f"-L{tmp_path}", "-l:libcudss.so.0"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return library


def _override(monkeypatch: pytest.MonkeyPatch, library: Path) -> None:
    monkeypatch.setenv("PHYDRAX_CUDA_CUDSS_LIBRARY", str(library))


@pytest.mark.parametrize("major", [11, 12, 14])
@pytest.mark.parametrize("platforms", [None, ("cpu",)])
def test_unsupported_plugin_refuses_before_backend_initialization(
    monkeypatch: pytest.MonkeyPatch,
    major: int,
    platforms: tuple[str, ...] | None,
) -> None:
    facts = CudaEnvironmentFacts(
        host_platform="linux",
        host_machine="x86_64",
        plugin_majors=(major,),
        selected_platforms=platforms,
    )
    monkeypatch.setattr(native, "_LOADED", {})
    monkeypatch.setattr(
        native, "collect_cuda_environment", lambda *, initialize_backend: facts
    )

    def forbidden_devices() -> list[jax.Device]:
        raise AssertionError("An unsupported plugin must not initialize the backend.")

    monkeypatch.setattr(jax, "local_devices", forbidden_devices)
    with pytest.raises(native.CudaNativeUnavailableError) as error:
        native.load_cuda_component("cudss")
    assert error.value.issue.code == "plugin-major-mismatch"
    assert error.value.issue.severity == "error"


def test_missing_companion_names_matching_extra(
    isolated_loader: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PHYDRAX_CUDA_CUDSS_LIBRARY", raising=False)
    monkeypatch.setattr(native.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(
        native.CudaNativeUnavailableError, match=r"phydrax\[cuda13\]"
    ) as error:
        native.load_cuda_component("cudss")
    assert error.value.issue.code == "companion-missing"
    assert isolated_loader == []


def test_missing_override_names_setting(
    isolated_loader: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _override(monkeypatch, tmp_path / "absent.so")
    with pytest.raises(
        native.CudaNativeUnavailableError,
        match="correct or unset PHYDRAX_CUDA_CUDSS_LIBRARY",
    ):
        native.load_cuda_component("cudss")
    assert isolated_loader == []


def test_provider_config_missing_override_names_setting(tmp_path: Path) -> None:
    missing = str(tmp_path / "missing-provider-override.so")
    source = f"""
from phydrax import config
from phydrax import _cuda_native as native
from phydrax._cuda_environment import CudaEnvironmentFacts
native._cuda_facts = lambda: CudaEnvironmentFacts(host_platform="linux", host_machine="x86_64", plugin_majors=(13,))
config.configure(config.ProviderConfig(library_paths={{"cuda_cudss": {missing!r}}}))
available, reason, issues, versions = native.cuda_component_status("cudss")
assert not available, reason
assert issues[0].code == "companion-missing", issues
assert "PHYDRAX_CUDA_CUDSS_LIBRARY" in reason, reason
assert {missing!r} in reason, reason
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("options", "code", "remedy"),
    [
        ({"version": "0.0.0"}, "companion-version-mismatch", "force-reinstall"),
        ({"major": 12}, "companion-major-mismatch", "CUDA 12"),
        ({"missing_handler": True}, "companion-invalid", "phydrax_cudss_release"),
        ({"omit_symbol": True}, "companion-invalid", "phx_cuda_shutdown"),
        ({"omit_handler_symbol": True}, "companion-invalid", "phydrax_cudss_release"),
        ({"dependency": True}, "native-dependency-missing", "nvidia-cudss-cu13"),
    ],
)
def test_native_refusals_precede_registration(
    isolated_loader: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    options: dict[str, Any],
    code: str,
    remedy: str,
) -> None:
    library = _compile_library(tmp_path, **options)
    _override(monkeypatch, library)
    with pytest.raises(native.CudaNativeUnavailableError) as error:
        native.load_cuda_component("cudss")
    assert error.value.issue.code == code
    assert remedy in str(error.value)
    assert "reinstall" in str(error.value).lower()
    assert isolated_loader == []


def test_registration_is_idempotent(
    isolated_loader: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library = _compile_library(tmp_path)
    _override(monkeypatch, library)
    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = tuple(pool.map(native.load_cuda_component, ("cudss",) * 4))
    first = identities[0]
    assert all(identity is first for identity in identities)
    assert native.load_cuda_component("cudss") is first
    assert len(isolated_loader) == 11
    assert len(set(isolated_loader)) == 11
    assert first.path == str(library)
    assert first.major == 13
    assert native.release_cuda_entries("cudss", 1, (1, 1, 0)) == 0
    resources = native.cuda_native_resources()
    assert resources.components == (first,)
    assert resources.partitions == ()


def test_failed_component_does_not_block_another(
    isolated_loader: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _override(monkeypatch, _compile_library(tmp_path, version="0.0.0"))
    assert not native.cuda_component_status("cudss")[0]
    amgx = _compile_library(tmp_path, component="amgx")
    monkeypatch.setenv("PHYDRAX_CUDA_AMGX_LIBRARY", str(amgx))
    assert native.load_cuda_component("amgx").name == "amgx"
    assert len(isolated_loader) == 4


def test_shutdown_drains_before_cleanup_and_logs_failures(
    isolated_loader: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _override(monkeypatch, _compile_library(tmp_path))
    monkeypatch.setenv(
        "PHYDRAX_CUDA_AMGX_LIBRARY",
        str(_compile_library(tmp_path, component="amgx")),
    )
    native.load_cuda_component("cudss")
    native.load_cuda_component("amgx")
    events: list[str] = []
    errors: list[str] = []

    def barrier() -> None:
        events.append("barrier")
        raise RuntimeError()

    def synchronize() -> None:
        events.append("synchronize")

    def record(event: str, status: int) -> Callable[..., int]:
        def call(*args: object) -> int:
            events.append(event)
            return status

        return call

    def emit(level: str, event: str, message: str, **fields: object) -> None:
        assert level == "ERROR"
        assert event == "cuda.native.shutdown_failed"
        assert message
        errors.append(message)

    monkeypatch.setattr(jax, "effects_barrier", barrier)
    monkeypatch.setattr(native, "_synchronize_devices", synchronize)
    monkeypatch.setattr(native, "emit", emit)
    for name, loaded in native._LOADED.items():
        loaded.functions["phx_cuda_release_all"] = record(
            f"{name}.release",
            3 if name == "cudss" else 0,
        )
        loaded.functions["phx_cuda_shutdown"] = record(f"{name}.shutdown", 0)
    native._shutdown()
    assert events == [
        "barrier",
        "synchronize",
        "cudss.release",
        "cudss.shutdown",
        "amgx.release",
        "amgx.shutdown",
    ]
    assert len(errors) == 2


def test_partition_identity_is_stable_and_positive() -> None:
    assert native.cuda_partition_key("plan-a") == native.cuda_partition_key("plan-a")
    assert native.cuda_partition_key("plan-a") != native.cuda_partition_key("plan-b")
    assert 0 < native.cuda_partition_key("plan-a") < 2**63


def test_traced_eager_release_names_traced_target() -> None:
    with pytest.raises(TypeError, match="phydrax_cudss_release"):
        jax.make_jaxpr(lambda ids: native.release_cuda_entries("cudss", 1, ids))(
            jax.ShapeDtypeStruct((1,), np.int32)
        )


def test_cpu_selected_process_never_loads_companion(tmp_path: Path) -> None:
    library = _compile_library(tmp_path)
    source = """
import ctypes
from phydrax._cuda_native import cuda_component_status
original = ctypes.CDLL
calls = []
def load(path, *args, **kwargs):
    calls.append(str(path))
    return original(path, *args, **kwargs)
ctypes.CDLL = load
available, reason, issues, versions = cuda_component_status("cudss")
assert not available, reason
assert issues and all(issue.severity == "error" for issue in issues), issues
assert not any("libfixture" in path for path in calls), calls
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[2],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "PHYDRAX_CUDA_CUDSS_LIBRARY": str(library),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.cuda_native
@pytest.mark.parametrize("component", ["cudss", "amgx", "quda"])
def test_real_component_registration_in_subprocess(component: str) -> None:
    source = f"""
from phydrax._cuda_native import load_cuda_component, CudaNativeUnavailableError
try:
    component = load_cuda_component({component!r})
except CudaNativeUnavailableError as error:
    print(str(error))
    raise SystemExit(77)
assert component.major == 13
assert component.path.startswith("/")
assert load_cuda_component({component!r}) is component
print(component)
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[2],
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    if result.returncode == 77:
        pytest.skip(result.stdout.strip())
    assert result.returncode == 0, result.stdout + result.stderr
