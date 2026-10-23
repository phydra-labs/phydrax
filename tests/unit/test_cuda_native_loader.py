"""Real C ABI fixtures exercise companion refusals before FFI registration."""

import ctypes
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NoReturn

import jax
import numpy as np
import pytest

from phydrax import _cuda_native as native
from phydrax._cuda_environment import CudaEnvironmentFacts, CudaPluginInstallation


@pytest.fixture
def isolated_loader(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    registrations: list[str] = []
    facts = CudaEnvironmentFacts(
        host_platform="linux",
        host_machine="x86_64",
        plugins=(
            CudaPluginInstallation(
                distribution="jax-cuda13-plugin", version="0.12.0", cuda_major=13
            ),
        ),
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
int32_t phx_cuda_release_partition(int64_t p, int64_t *released) {{ *released=0; return 0; }}
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
        plugins=(
            CudaPluginInstallation(
                distribution=f"jax-cuda{major}-plugin", version="0.12.0", cuda_major=major
            ),
        ),
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
from phydrax._cuda_environment import CudaEnvironmentFacts, CudaPluginInstallation
native._cuda_facts = lambda: CudaEnvironmentFacts(host_platform="linux", host_machine="x86_64", plugins=(CudaPluginInstallation(distribution="jax-cuda13-plugin", version="0.12.0", cuda_major=13),))
config.configure(config.ProviderConfig(library_paths={{"cuda_cudss": {missing!r}}}))
available, reason, issues, versions = native.cuda_component_status("cudss")
assert not available, reason
assert issues[0].code == "companion-missing", issues
assert "PHYDRAX_CUDA_CUDSS_LIBRARY" in reason, reason
assert issues[0].paths == ({missing!r},), issues[0].paths
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


@pytest.mark.parametrize("failed_dispatch", [False, True])
def test_partition_release_drains_even_failed_dispatch_without_token_reads(
    isolated_loader: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failed_dispatch: bool,
) -> None:
    _override(monkeypatch, _compile_library(tmp_path))
    native.load_cuda_component("cudss")
    events: list[str] = []

    def barrier() -> None:
        events.append("barrier")
        if failed_dispatch:
            raise RuntimeError("failed preparation")

    def synchronize() -> None:
        events.append("synchronize")

    def release(partition: int, output: object) -> int:
        assert partition == 17
        events.append("release")
        return 0

    def forbidden_get(value: object) -> object:
        raise AssertionError("partition release must not materialize device tokens")

    monkeypatch.setattr(jax, "effects_barrier", barrier)
    monkeypatch.setattr(jax, "device_get", forbidden_get)
    monkeypatch.setattr(native, "_synchronize_devices", synchronize)
    native._LOADED["cudss"].functions["phx_cuda_release_partition"] = release
    if failed_dispatch:
        with pytest.raises(RuntimeError, match="failed preparation"):
            native.release_cuda_partition("cudss", 17)
    else:
        assert native.release_cuda_partition("cudss", 17) == 0
    assert events == ["barrier", "synchronize", "release"]


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
from pathlib import Path
from phydrax._cuda_native import load_cuda_component, CudaNativeUnavailableError
try:
    component = load_cuda_component({component!r})
except CudaNativeUnavailableError as error:
    print(str(error))
    raise SystemExit(77)
assert component.major == 13
assert Path(component.path).is_absolute()
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


def _windows_sync_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    synchronize_status: int = 0,
    restore_status: int = 0,
    get_status: int = 0,
    set_status: int = 0,
    raise_synchronize: bool = False,
) -> list[tuple[str, int]]:
    events: list[tuple[str, int]] = []

    def get_device(output: Any) -> int:
        events.append(("get", 7))
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int32))[0] = 7
        return get_status

    def set_device(device: int) -> int:
        events.append(("set", device))
        return restore_status if device == 7 else set_status

    def synchronize() -> int:
        events.append(("synchronize", 0))
        if raise_synchronize:
            raise RuntimeError("synchronization invocation failed")
        return synchronize_status

    api = SimpleNamespace(
        cudaGetDevice=get_device,
        cudaSetDevice=set_device,
        cudaDeviceSynchronize=synchronize,
    )
    identity = native.CudaNativeComponent(
        name="cudss",
        path="admitted.dll",
        version="0.2.2",
        build_hash="a" * 64,
        major=13,
        preloaded=("admitted.dll",),
    )
    monkeypatch.setattr(
        native,
        "_LOADED",
        {"cudss": native._Loaded(identity, ctypes.CDLL(None, handle=0), {}, (), api)},
    )
    monkeypatch.setattr(native, "sys", SimpleNamespace(platform="win32"))
    module = ModuleType("phydrax_cuda13_pjrt.runtime")
    monkeypatch.setattr(module, "runtime_identity", lambda: None, raising=False)
    monkeypatch.setattr(
        module,
        "cuda_runtime_api",
        lambda *_: pytest.fail("Shutdown must use its retained ABI"),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "load_libraries",
        lambda *_: pytest.fail("Shutdown must not load a DLL"),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(
        sys.modules, "phydrax_cuda13_pjrt", ModuleType("phydrax_cuda13_pjrt")
    )
    monkeypatch.setattr(
        jax, "local_devices", lambda: [SimpleNamespace(local_hardware_id=0)]
    )
    monkeypatch.setattr(native, "device_platform", lambda _: "cuda")
    return events


@pytest.mark.parametrize(
    "failure,expected",
    [
        ({}, None),
        ({"synchronize_status": 9}, "synchronization failed.*status 9"),
        ({"set_status": 8}, "synchronization failed.*status 8"),
        ({"restore_status": 7}, "restoration failed.*status 7"),
        ({"raise_synchronize": True}, "synchronization invocation failed"),
    ],
)
def test_windows_sync_uses_owning_runtime_and_restores_caller_device(
    monkeypatch: pytest.MonkeyPatch, failure: dict[str, Any], expected: str | None
) -> None:
    events = _windows_sync_state(monkeypatch, **failure)
    if expected is None:
        native._synchronize_devices()
    else:
        with pytest.raises(RuntimeError, match=expected):
            native._synchronize_devices()
    assert events[:2] == [("get", 7), ("set", 0)]
    assert events[-1] == ("set", 7)
    assert (("synchronize", 0) in events) is not bool(failure.get("set_status"))


def test_windows_sync_failed_current_device_query_never_changes_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _windows_sync_state(monkeypatch, get_status=42)
    with pytest.raises(RuntimeError, match="caller device query failed.*status 42"):
        native._synchronize_devices()
    assert events == [("get", 7)]


def test_windows_sync_preserves_both_operation_and_restore_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _windows_sync_state(monkeypatch, synchronize_status=9, restore_status=7)
    with pytest.raises(RuntimeError, match="status 9.*restoration failed.*status 7"):
        native._synchronize_devices()


def test_windows_identity_observation_failure_does_not_prevent_device_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _windows_sync_state(monkeypatch)
    failure = OSError("module inventory refused")

    def refuse_identity() -> NoReturn:
        raise failure

    monkeypatch.setattr(
        sys.modules["phydrax_cuda13_pjrt.runtime"], "runtime_identity", refuse_identity
    )
    monkeypatch.setattr(
        jax,
        "local_devices",
        lambda: [
            SimpleNamespace(local_hardware_id=0),
            SimpleNamespace(local_hardware_id=1),
        ],
    )
    with pytest.raises((RuntimeError, OSError)) as caught:
        native._synchronize_devices()
    assert events == [
        ("get", 7),
        ("set", 0),
        ("synchronize", 0),
        ("set", 1),
        ("synchronize", 0),
        ("set", 7),
    ]
    assert str(failure) in str(caught.value)


@pytest.mark.parametrize("state", ("empty", "no-dependencies", "missing-abi"))
def test_windows_empty_or_partial_shutdown_never_loads_another_component(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    events = _windows_sync_state(monkeypatch)
    loaded = native._LOADED["cudss"]
    preloaded = () if state == "no-dependencies" else loaded.identity.preloaded
    identity = native.CudaNativeComponent(
        name="cudss",
        path="admitted.dll",
        version="0.2.2",
        build_hash="a" * 64,
        major=13,
        preloaded=preloaded,
    )
    monkeypatch.setattr(
        native,
        "_LOADED",
        {}
        if state == "empty"
        else {"cudss": native._Loaded(identity, ctypes.CDLL(None, handle=0), {}, ())},
    )
    monkeypatch.setattr(
        native, "load_cuda_component", lambda *_: pytest.fail("Unexpected component load")
    )
    monkeypatch.setattr(
        jax, "local_devices", lambda: pytest.fail("Unexpected device query")
    )
    if state == "missing-abi":
        with pytest.raises(RuntimeError, match="owning runtime ABI was not bound"):
            native._synchronize_devices()
    else:
        native._synchronize_devices()
    assert events == []


@pytest.mark.parametrize("platform", ("win32", "linux"))
def test_driver_hybrid_compatibility_is_not_compiled_cuda_major(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
) -> None:
    facts = SimpleNamespace(plugins=(object(),))
    monkeypatch.setattr(native, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(native, "collect_cuda_environment", lambda **_: facts)
    monkeypatch.setattr(
        native, "assess_cuda_environment", lambda _: SimpleNamespace(issues=())
    )
    monkeypatch.setattr(native, "device_platform", lambda _: "cuda")
    monkeypatch.setattr(
        jax,
        "local_devices",
        lambda: [SimpleNamespace(client=SimpleNamespace(platform_version="cuda 14010"))],
    )
    if platform == "win32":
        assert native._cuda_facts() is facts
    else:
        with pytest.raises(native.CudaNativeUnavailableError, match="not compatible"):
            native._cuda_facts()


def test_windows_cleanup_drains_each_components_own_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _windows_sync_state(monkeypatch)

    def get_device(output: Any) -> int:
        events.append(("amgx.get", 3))
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int32))[0] = 3
        return 0

    def set_device(device: int) -> int:
        events.append(("amgx.set", device))
        return 0

    def synchronize() -> int:
        events.append(("amgx.synchronize", 0))
        return 0

    identity = native.CudaNativeComponent(
        name="amgx",
        path="amgx.dll",
        version="0.2.2",
        build_hash="a" * 64,
        major=13,
        preloaded=("amgx.dll",),
    )
    native._LOADED["amgx"] = native._Loaded(
        identity,
        ctypes.CDLL(None, handle=0),
        {},
        (),
        SimpleNamespace(
            cudaGetDevice=get_device,
            cudaSetDevice=set_device,
            cudaDeviceSynchronize=synchronize,
        ),
    )
    native._synchronize_devices()
    assert events == [
        ("get", 7),
        ("set", 0),
        ("synchronize", 0),
        ("set", 7),
        ("amgx.get", 3),
        ("amgx.set", 0),
        ("amgx.synchronize", 0),
        ("amgx.set", 3),
    ]
