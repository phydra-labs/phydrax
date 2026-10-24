from __future__ import annotations

import base64
import ctypes
import hashlib
import importlib.util
import json
import os
import sys
from importlib import metadata
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NoReturn, TypedDict

import pytest


class _PayloadEntry(TypedDict):
    path: str
    sha256: str
    size: int
    mode: int


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    source = (
        Path(__file__).resolve().parents[2]
        / "packaging/rocm/python/phydrax_rocm_runtime/_loader.py"
    )
    spec = importlib.util.spec_from_file_location("_phydrax_rocm_loader_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(module, "_loaded_paths", lambda: ())
    monkeypatch.delitem(sys.modules, "jax_rocm10_plugin", raising=False)
    monkeypatch.delitem(sys.modules, "jax_plugins.xla_rocm10", raising=False)
    for name in ("LLVM_PATH", "ROCM_PATH", "ROCM_HOME", "HIP_DEVICE_LIB_PATH"):
        monkeypatch.delenv(name, raising=False)
    return module


@pytest.fixture
def installed(
    runtime: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    distributions: dict[str, metadata.Distribution] = {}
    inventory: dict[str, list[_PayloadEntry]] = {}

    def payload(path: str, data: bytes = b"owned", mode: int = 0o644) -> _PayloadEntry:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(mode)
        return {
            "path": path,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "mode": mode,
        }

    def distribution(
        name: str,
        version: str,
        entries: list[_PayloadEntry],
        extra: dict[str, str] | None = None,
    ) -> None:
        info = tmp_path / f"{name.replace('-', '_')}-{version}.dist-info"
        info.mkdir(exist_ok=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
        )
        for filename, content in (extra or {}).items():
            (info / filename).write_text(content)
        rows = []
        for entry in entries:
            digest = (
                base64.urlsafe_b64encode(bytes.fromhex(entry["sha256"]))
                .decode()
                .rstrip("=")
            )
            rows.append(f"{entry['path']},sha256={digest},{entry['size']}")
        (info / "RECORD").write_text("\n".join(rows) + "\n")
        distributions[name] = metadata.PathDistribution(info)

    core = [
        payload("_rocm_sdk_core/lib/libhsa-runtime64.so.1", b"base library"),
        payload("_rocm_sdk_core/lib/llvm/bin/ld.lld", b"linker", 0o755),
        payload("_rocm_sdk_core/lib/llvm/amdgcn/bitcode/ocml.bc", b"bitcode"),
        payload("rocm_sdk_core/__init__.py", b"upstream SDK helper"),
        payload("_rocm_sdk_core/share/therock/therock_manifest.json", b"{}"),
    ]
    libraries = [
        payload("_rocm_sdk_libraries/lib/libamdhip64.so.6", b"hip library"),
        payload("rocm_sdk_libraries/__init__.py", b"upstream library helper"),
    ]
    inventory["core"] = core
    inventory["libraries"] = libraries
    for arch in runtime.ARCHITECTURES:
        inventory[f"device_{arch}"] = [
            payload(f"_rocm_sdk_libraries/.kpack/rocblas_{arch}.kpack", arch.encode())
        ]
    components = {}
    for key, (name, module) in runtime.COMPONENTS.items():
        entries = sorted(inventory[key], key=lambda entry: entry["path"])
        encoded = json.dumps(entries, separators=(",", ":"), ensure_ascii=True)
        distribution(name, "10.1.0", entries, {"phydrax-payload.json": encoded})
        components[key] = {
            "distribution": name,
            "version": "10.1.0",
            "module": module,
            "upstream": {"url": f"https://repo.amd.com/{key}.whl", "sha256": "a" * 64},
            "payload_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "deduplicated": [],
        }
    native = [
        {
            "path": core[0]["path"],
            "soname": "libhsa-runtime64.so.1",
            "needed": ["libc.so.6"],
            "sha256": core[0]["sha256"],
        },
        {
            "path": libraries[0]["path"],
            "soname": "libamdhip64.so.6",
            "needed": ["libhsa-runtime64.so.1"],
            "sha256": libraries[0]["sha256"],
        },
    ]
    plugin = payload("jax_rocm10_plugin/_solver.so", b"solver")
    distribution("jax-rocm10-plugin", "0.12.0", [plugin])
    for name in ("jax", "jaxlib", "jax-rocm10-pjrt"):
        distribution(name, "0.12.0", [])
    manifest = {
        "sdk_version": "10.1.0",
        "architectures": ["gfx1250", "gfx942", "gfx950"],
        "components": components,
        "source_manifest": core[-1]["path"],
        "third_party_identity": "AMD ROCm SDK 10.1.0",
        "packaging_identity": "Phydrax owned ROCm runtime packaging",
        "loader": {
            "libraries": native,
            "preload": [entry["path"] for entry in native],
            "compiler": {
                "rocm_path": "_rocm_sdk_core",
                "llvm_path": "_rocm_sdk_core/lib/llvm",
                "lld_path": core[1]["path"],
                "bitcode_path": "_rocm_sdk_core/lib/llvm/amdgcn/bitcode",
            },
            "kpack": {
                arch: [inventory[f"device_{arch}"][0]["path"]]
                for arch in runtime.ARCHITECTURES
            },
            "system_libraries": sorted(runtime._SYSTEM_LIBRARIES),
            "plugin_objects": [
                {
                    "wheel": "jax_rocm10_plugin-0.12.0.whl",
                    "path": plugin["path"],
                    "soname": None,
                    "needed": ["libamdhip64.so.6"],
                    "sha256": plugin["sha256"],
                }
            ],
        },
    }

    def save() -> None:
        entry = payload(
            "phydrax_rocm_runtime/runtime.json", json.dumps(manifest).encode()
        )
        distribution("phydrax-rocm-runtime", "10.1.0", [entry])

    save()

    def lookup(name: str) -> metadata.Distribution:
        if name not in distributions:
            raise metadata.PackageNotFoundError(name)
        return distributions[name]

    monkeypatch.setattr(runtime.metadata, "distribution", lookup)
    monkeypatch.setattr(
        runtime.metadata, "distributions", lambda: tuple(distributions.values())
    )
    return SimpleNamespace(
        root=tmp_path,
        manifest=manifest,
        save=save,
        distributions=distributions,
        inventory=inventory,
    )


def test_inspection_is_pure_and_preserves_architecture_identity(
    runtime: ModuleType, installed: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Inspection must not dlopen.")

    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    before = dict(os.environ)
    bundle = runtime.inspect_rocm_bundle()
    assert bundle.sdk_version == "10.1.0"
    assert bundle.architectures == ("gfx1250", "gfx942", "gfx950")
    assert len(bundle.components) == 5
    assert not runtime.rocm_bundle_loaded()
    assert dict(os.environ) == before


def test_component_cannot_claim_another_sdk_namespace(
    runtime: ModuleType, installed: SimpleNamespace
) -> None:
    relative = "rocm_sdk_libraries/foreign.py"
    content = b"not a core SDK payload"
    (installed.root / relative).write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    entry = {
        "path": relative,
        "sha256": digest,
        "size": len(content),
        "mode": 0o644,
    }
    entries = sorted([*installed.inventory["core"], entry], key=lambda item: item["path"])
    encoded = json.dumps(entries, separators=(",", ":"), ensure_ascii=True)
    info = installed.root / "phydrax_rocm_core-10.1.0.dist-info"
    (info / "phydrax-payload.json").write_text(encoded)
    record = info / "RECORD"
    wheel_digest = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip("=")
    record.write_text(
        record.read_text() + f"{relative},sha256={wheel_digest},{len(content)}\n"
    )
    installed.manifest["components"]["core"]["payload_sha256"] = hashlib.sha256(
        encoded.encode()
    ).hexdigest()
    installed.save()
    with pytest.raises(runtime.RocmRuntimeError):
        runtime.inspect_rocm_bundle()


def test_missing_bundle_is_optional(
    runtime: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(name: str) -> NoReturn:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runtime.metadata, "distribution", missing)
    assert runtime.inspect_rocm_bundle() is None


@pytest.mark.parametrize(
    "field,value",
    [("sdk_version", 10), ("sdk_version", "6.4.4"), ("architectures", ["gfx942"])],
)
def test_manifest_identity_is_exact(
    runtime: ModuleType,
    installed: SimpleNamespace,
    field: str,
    value: int | str | list[str],
) -> None:
    installed.manifest[field] = value
    installed.save()
    with pytest.raises(runtime.RocmRuntimeError):
        runtime.inspect_rocm_bundle()


def test_manifest_record_is_checked(
    runtime: ModuleType, installed: SimpleNamespace
) -> None:
    path = installed.root / "phydrax_rocm_runtime/runtime.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(runtime.RocmRuntimeError, match="RECORD"):
        runtime.inspect_rocm_bundle()


def test_inventory_digest_and_versions_are_checked(
    runtime: ModuleType, installed: SimpleNamespace
) -> None:
    installed.manifest["components"]["core"]["payload_sha256"] = "0" * 64
    installed.save()
    with pytest.raises(runtime.RocmRuntimeError, match="inventory identity"):
        runtime.inspect_rocm_bundle()


def test_architecture_asset_cannot_escape_installation(
    runtime: ModuleType, installed: SimpleNamespace, tmp_path: Path
) -> None:
    path = installed.root / installed.manifest["loader"]["kpack"]["gfx950"][0]
    outside = tmp_path.parent / f"{tmp_path.name}-outside.kpack"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(runtime.RocmRuntimeError, match="escapes"):
        runtime.inspect_rocm_bundle()


def test_dependency_order_is_validated(
    runtime: ModuleType, installed: SimpleNamespace
) -> None:
    installed.manifest["loader"]["preload"].reverse()
    installed.save()
    with pytest.raises(runtime.RocmRuntimeError, match="dependency order"):
        runtime.inspect_rocm_bundle()


def test_foreign_distribution_is_refused(
    runtime: ModuleType, installed: SimpleNamespace
) -> None:
    installed.distributions["rocm-sdk-core"] = SimpleNamespace(
        metadata={"Name": "rocm-sdk-core"}, version="10.1.0"
    )
    with pytest.raises(runtime.RocmRuntimeError, match="Conflicting native distribution"):
        runtime.inspect_rocm_bundle()


def test_activation_uses_absolute_dependency_order_retains_handles_and_no_ld_path(
    runtime: ModuleType, installed: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    if sys.platform == "win32":
        pytest.skip("ROCm preload activation requires POSIX RTLD loader flags.")
    expected_mode = os.RTLD_NOW | os.RTLD_GLOBAL
    calls = []
    handles = []

    def load(path: str, *, mode: int) -> object:
        assert Path(path).is_absolute()
        assert mode == expected_mode
        calls.append(path)
        handle = object()
        handles.append(handle)
        return handle

    monkeypatch.setattr(ctypes, "CDLL", load)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/foreign/not-used")
    bundle = runtime.activate_rocm_bundle()
    assert calls == list(bundle.preload)
    assert runtime._HANDLES == handles
    assert os.environ["LD_LIBRARY_PATH"] == "/foreign/not-used"
    compiler = dict(bundle.compiler)
    assert os.environ["LLVM_PATH"] == compiler["llvm_path"]
    assert os.environ["ROCM_PATH"] == compiler["rocm_path"]
    assert os.environ["HIP_DEVICE_LIB_PATH"] == compiler["bitcode_path"]

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError(
            "A bound runtime must not re-read inventories or hash payloads."
        )

    monkeypatch.setattr(runtime, "_inspect", forbidden)
    monkeypatch.setattr(runtime.hashlib, "file_digest", forbidden)
    assert runtime.inspect_rocm_bundle() is bundle
    assert runtime.activate_rocm_bundle() is bundle
    assert len(calls) == len(bundle.preload)


def test_foreign_loaded_runtime_is_refused_before_loading(
    runtime: ModuleType, installed: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runtime, "_loaded_paths", lambda: (Path("/opt/rocm/lib/libamdhip64.so.6"),)
    )
    with pytest.raises(runtime.RocmRuntimeError, match="Foreign ROCm"):
        runtime.activate_rocm_bundle()
    assert runtime._HANDLES == []


@pytest.mark.parametrize("kind", ["library", "linker", "gfx1250", "gfx942", "gfx950"])
def test_same_size_payload_tampering_is_refused(
    runtime: ModuleType, installed: SimpleNamespace, kind: str
) -> None:
    loader = installed.manifest["loader"]
    if kind == "library":
        relative = loader["libraries"][0]["path"]
    elif kind == "linker":
        relative = loader["compiler"]["lld_path"]
    else:
        relative = loader["kpack"][kind][0]
    path = installed.root / relative
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(runtime.RocmRuntimeError, match="digest mismatch"):
        runtime.activate_rocm_bundle()
    assert runtime._HANDLES == []


def test_late_import_and_compiler_override_are_refused(
    runtime: ModuleType, installed: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLVM_PATH", "/opt/rocm/llvm")
    with pytest.raises(runtime.RocmRuntimeError, match="LLVM_PATH conflicts"):
        runtime.activate_rocm_bundle()
    assert runtime._HANDLES == []


def test_late_plugin_is_refused(
    runtime: ModuleType, installed: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "jax_rocm10_plugin", SimpleNamespace())
    with pytest.raises(runtime.RocmRuntimeError, match="before the owned loader"):
        runtime.activate_rocm_bundle()


def test_finder_targets_only_exact_rocm_provider_modules(
    runtime: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        runtime, "activate_rocm_bundle", lambda **kwargs: calls.append("activate")
    )
    original_loader = SimpleNamespace(
        create_module=lambda spec: None,
        exec_module=lambda module: calls.append("execute"),
    )
    monkeypatch.setattr(
        runtime.importlib.machinery.PathFinder,
        "find_spec",
        lambda fullname, path, target: runtime.importlib.machinery.ModuleSpec(
            fullname, original_loader
        ),
    )
    finder = runtime._RocmFinder()
    for name in (
        "jax",
        "jax.numpy",
        "jax_plugins",
        "jax_plugins.xla_cuda13",
        "jax_rocm10_plugin.version",
    ):
        assert finder.find_spec(name) is None
    assert calls == []
    for name in ("jax_rocm10_plugin", "jax_plugins.xla_rocm10"):
        spec = finder.find_spec(name)
        assert isinstance(spec.loader, runtime._RocmLoader)
        assert spec.loader.create_module(spec) is None
        # Finding a module is inspection; activation precedes its execution only.
        assert calls == []
        spec.loader.exec_module(SimpleNamespace())
        assert calls == ["activate", "execute"]
        calls.clear()


def test_hook_installation_does_not_activate(
    runtime: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = list(sys.meta_path)
    monkeypatch.setattr(sys, "meta_path", original.copy())
    runtime.install_rocm_loader()
    runtime.install_rocm_loader()
    assert len(sys.meta_path) == len(original) + 1
    assert runtime._HANDLES == []
    assert not runtime.rocm_bundle_loaded()


def test_missing_architecture_payload_is_refused(
    runtime: ModuleType, installed: SimpleNamespace
) -> None:
    path = installed.root / installed.manifest["loader"]["kpack"]["gfx950"][0]
    path.unlink()
    with pytest.raises(runtime.RocmRuntimeError):
        runtime.inspect_rocm_bundle()


def test_partial_load_retains_handles_and_refuses_retry(
    runtime: ModuleType, installed: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    retained = object()

    def load(path: str, *, mode: int) -> object:
        calls.append(path)
        if len(calls) == 2:
            raise OSError("unavailable host ABI")
        return retained

    monkeypatch.setattr(ctypes, "CDLL", load)
    with pytest.raises(runtime.RocmRuntimeError, match="unavailable host ABI"):
        runtime.activate_rocm_bundle()
    assert runtime._HANDLES == [retained]
    assert runtime.initial_rocm_environment() == ()
    with pytest.raises(runtime.RocmRuntimeError, match="unavailable host ABI"):
        runtime.activate_rocm_bundle()
    assert len(calls) == 2
