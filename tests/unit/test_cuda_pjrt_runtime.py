"""RECORD-backed ownership/admission tests; PE fixtures are never executed."""

from __future__ import annotations

import base64
import csv
import ctypes
import functools
import hashlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NoReturn

import pytest

from tests._support.pe import write_pe


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "native/cuda13_pjrt/python/phydrax_cuda13_pjrt"


def _record(
    root: Path,
    name: str,
    version: str,
    files: list[str],
    *,
    requirements: tuple[str, ...] = (),
) -> metadata.Distribution:
    info = root / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
        + "".join(f"Requires-Dist: {item}\n" for item in requirements),
        encoding="utf-8",
        newline="\n",
    )
    rows = []
    for relative in [*files, (info / "METADATA").relative_to(root).as_posix()]:
        data = (root / relative).read_bytes()
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        )
        rows.append((relative, "sha256=" + digest, str(len(data))))
    rows.append(((info / "RECORD").relative_to(root).as_posix(), "", ""))
    output = io.StringIO(newline="")
    csv.writer(output).writerows(rows)
    (info / "RECORD").write_text(output.getvalue(), encoding="utf-8", newline="")
    return metadata.PathDistribution(info)


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # The released wheel receives this exact canonical parser through staging.
    package = tmp_path / "isolated_runtime"
    shutil.copytree(RUNTIME, package)
    shutil.copyfile(ROOT / "phydrax/_pe_dependencies.py", package / "_pe_dependencies.py")
    name = "isolated_cuda_runtime"
    spec = importlib.util.spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    child = importlib.util.spec_from_file_location(
        f"{name}.runtime", package / "runtime.py"
    )
    assert child is not None and child.loader is not None
    result = importlib.util.module_from_spec(child)
    monkeypatch.setitem(sys.modules, child.name, result)
    child.loader.exec_module(result)
    return result


@pytest.fixture
def installed(
    runtime: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, metadata.Distribution]]:
    root = tmp_path / "installed Ω with spaces"
    pjrt = write_pe(
        root / "phydrax_cuda13_pjrt/lib/pjrt.dll",
        imports=("CuBlAs64_13.DLL",),
        exports=(("GetPjrtApi", 1, None),),
    )
    kernel = write_pe(
        root / "jax_cuda13_plugin/_solver.pyd", exports=(("PyInit__solver", 1, None),)
    )
    cublas = write_pe(root / "nvidia/cu13/bin/x86_64/cublas64_13.dll")
    assets = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
            "role": role,
        }
        for path, role in ((pjrt, "pjrt"), (kernel, "kernel"))
    ]
    record: dict[str, Any] = {
        "distribution": runtime.DISTRIBUTION,
        "version": "0.2.2",
        "cuda_major": 13,
        "jax_version": "0.12.0",
        "jaxlib_version": "0.12.0",
        "python_tag": "cp313",
        "abi_tag": "cp313",
        "platform_tag": "win_amd64",
        "capabilities": ["xla_gpu"],
        "assets": assets,
    }
    record["build_id"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = root / "phydrax_cuda13_pjrt/manifest.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    distributions = {
        runtime.DISTRIBUTION: _record(
            root,
            runtime.DISTRIBUTION,
            "0.2.2",
            [path.relative_to(root).as_posix() for path in (pjrt, kernel, manifest)],
            requirements=("nvidia-cublas==13.4.92",),
        ),
        "nvidia-cublas": _record(
            root, "nvidia-cublas", "13.4.92", [cublas.relative_to(root).as_posix()]
        ),
        "jax": _record(root, "jax", "0.12.0", []),
        "jaxlib": _record(root, "jaxlib", "0.12.0", []),
    }

    def distribution(name: str) -> metadata.Distribution:
        if name not in distributions:
            raise metadata.PackageNotFoundError(name)
        return distributions[name]

    monkeypatch.setattr(runtime.metadata, "distribution", distribution)
    monkeypatch.setattr(
        runtime.metadata, "version", lambda name: distribution(name).version
    )
    monkeypatch.setattr(
        runtime.metadata, "distributions", lambda: iter(distributions.values())
    )
    monkeypatch.setattr(runtime, "_mapped_paths", lambda: (str(root / "python.exe"),))
    system = root / "Windows/System32"
    system.mkdir(parents=True)
    monkeypatch.setattr(runtime, "_system_directory", lambda: system)
    return root, distributions


def test_installation_record_is_metadata_only(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> None:
        raise AssertionError("Metadata admission invoked a native observer")

    monkeypatch.setattr(runtime, "_mapped_paths", forbidden)
    monkeypatch.setattr(runtime, "_kernel32", forbidden)
    record = runtime.installation_record()
    assert (
        record.cuda_major == 13
        and record.jax_version == record.jaxlib_version == "0.12.0"
    )
    assert record.capabilities == ("xla_gpu",)
    assert len(record.assets) == 2 and record.issues == ()


@pytest.mark.parametrize(
    "owner", ["phydrax-cuda13-pjrt", "phydrax-cuda13", "nvidia-cublas"]
)
def test_missing_distribution_metadata_refuses_before_native_loading(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    owner: str,
) -> None:
    root, distributions = installed
    if owner == "phydrax-cuda13":
        distributions[owner] = _record(root, owner, "0.2.2", [])
    dist = distributions[owner]
    members = dist.files
    assert members is not None
    metadata_path = next(path for path in members if path.name == "METADATA")
    (root / metadata_path).unlink()
    with pytest.raises(RuntimeError, match=f"metadata for {owner}"):
        runtime.dependency_closure(root / "phydrax_cuda13_pjrt/lib/pjrt.dll")


def test_mismatched_companion_release_refuses_before_native_loading(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
) -> None:
    root, distributions = installed
    distributions["phydrax-cuda13"] = _record(root, "phydrax-cuda13", "0.0.0", [])
    with pytest.raises(
        RuntimeError, match=r"requires phydrax-cuda13==0\.2\.2, found version 0\.0\.0"
    ):
        runtime.dependency_closure(root / "phydrax_cuda13_pjrt/lib/pjrt.dll")


@pytest.mark.parametrize(
    "fault", [None, "missing-linker", "corrupt-linker", "split-root"]
)
def test_compiler_admission_requires_owned_assembler_linker_and_libdevice(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
    fault: str | None,
) -> None:
    root, distributions = installed
    owner = runtime.DISTRIBUTION
    members = distributions[owner].files
    assert members is not None
    files = [item.as_posix() for item in members if ".dist-info/" not in item.as_posix()]
    compiler = root / "phydrax_cuda13_pjrt/cuda"
    linker = compiler / ("other" if fault == "split-root" else "bin") / "nvlink.exe"
    for path in (
        compiler / "bin/ptxas.exe",
        linker,
        compiler / "nvvm/libdevice/libdevice.10.bc",
    ):
        if path == linker and fault == "missing-linker":
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"authenticated compiler fixture")
        files.append(path.relative_to(root).as_posix())
    distributions[owner] = _record(
        root, owner, "0.2.2", files, requirements=("nvidia-cublas==13.4.92",)
    )
    if fault == "corrupt-linker":
        linker.write_bytes(b"tampered")
    monkeypatch.setenv("PATH", str(root / "unrelated-tools"))
    if fault is None:
        assert Path(runtime.compiler_root()) == compiler
        assert os.environ["PATH"].split(os.pathsep) == [
            str(compiler / "bin"),
            str(root / "unrelated-tools"),
        ]
    else:
        message = {
            "missing-linker": "Missing RECORD-owned compiler assets.*nvlink",
            "corrupt-linker": "integrity failure",
            "split-root": "required CUDA data root",
        }[fault]
        with pytest.raises(RuntimeError, match=message):
            runtime.compiler_root()


def test_closure_resolves_case_insensitive_record_owned_imports(
    runtime: ModuleType, installed: tuple[Path, dict[str, metadata.Distribution]]
) -> None:
    root, _ = installed
    closure = runtime.dependency_closure(root / "phydrax_cuda13_pjrt/lib/pjrt.dll")
    assert closure.complete and closure.issues == ()
    assert [Path(item.path).name for item in closure.objects] == [
        "cublas64_13.dll",
        "pjrt.dll",
    ]
    assert [(edge.name, edge.owner, edge.kind) for edge in closure.edges] == [
        ("CuBlAs64_13.DLL", "nvidia-cublas", "owned")
    ]


def test_corrupt_asset_and_record_refuse(
    runtime: ModuleType, installed: tuple[Path, dict[str, metadata.Distribution]]
) -> None:
    root, _ = installed
    (root / "phydrax_cuda13_pjrt/lib/pjrt.dll").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="integrity"):
        runtime.installation_record()
    with pytest.raises(RuntimeError, match="integrity"):
        runtime.dependency_closure(root / "phydrax_cuda13_pjrt/lib/pjrt.dll")


def test_missing_manifest_is_not_an_assumed_capability(
    runtime: ModuleType, installed: tuple[Path, dict[str, metadata.Distribution]]
) -> None:
    root, _ = installed
    (root / "phydrax_cuda13_pjrt/manifest.json").unlink()
    with pytest.raises(
        RuntimeError, match=r"Missing or duplicate RECORD ownership.*manifest\.json"
    ):
        runtime.installation_record()


def test_competing_kernel_owner_refuses(
    runtime: ModuleType, installed: tuple[Path, dict[str, metadata.Distribution]]
) -> None:
    root, distributions = installed
    distributions["foreign-plugin"] = _record(
        root, "foreign-plugin", "0.12.0", ["jax_cuda13_plugin/_solver.pyd"]
    )
    with pytest.raises(RuntimeError, match="ownership"):
        runtime.installation_record()


def test_ambiguous_dll_basename_refuses(
    runtime: ModuleType, installed: tuple[Path, dict[str, metadata.Distribution]]
) -> None:
    root, distributions = installed
    duplicate = write_pe(root / "jaxlib/cublas64_13.dll")
    distributions["jaxlib"] = _record(
        root, "jaxlib", "0.12.0", [duplicate.relative_to(root).as_posix()]
    )
    with pytest.raises(RuntimeError, match="Ambiguous.*basename"):
        runtime.dependency_closure(root / "phydrax_cuda13_pjrt/lib/pjrt.dll")


@pytest.mark.parametrize(
    "foreign_resident", [False, True], ids=["unmapped-stable-abi", "foreign-resident"]
)
def test_python_stable_and_minor_abi_libraries_require_interpreter_ownership(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
    foreign_resident: bool,
) -> None:
    root, distributions = installed
    interpreter = root / "managed-python"
    stable = write_pe(interpreter / "python3.dll", imports=("python313.dll",))
    minor = write_pe(interpreter / "python313.dll")
    extension = write_pe(
        root / "jaxlib/abi-test.pyd", imports=("python3.dll", "python313.dll")
    )
    distributions["jaxlib"] = _record(
        root, "jaxlib", "0.12.0", [extension.relative_to(root).as_posix()]
    )
    monkeypatch.setattr(runtime.sys, "base_prefix", str(interpreter))
    monkeypatch.setattr(runtime.sys, "executable", str(interpreter / "python.exe"))
    mapped = [str(minor)]
    if foreign_resident:
        mapped.append(str(write_pe(root / "foreign/python3.dll")))
    monkeypatch.setattr(runtime, "_mapped_paths", lambda: tuple(mapped))
    if foreign_resident:
        with pytest.raises(RuntimeError):
            runtime.dependency_closure(extension)
        return
    closure = runtime.dependency_closure(extension)
    assert closure.complete, closure.issues
    edges = {edge.name.casefold(): edge for edge in closure.edges}
    assert edges["python3.dll"].owner == "CPython"
    assert edges["python3.dll"].path == str(stable.resolve())
    assert edges["python3.dll"].kind == "system"
    assert edges["python313.dll"].owner == "CPython"
    assert edges["python313.dll"].path == str(minor.resolve())
    assert edges["python313.dll"].kind == "resident"


def test_hostile_resident_family_refuses_before_loading(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = installed
    monkeypatch.setattr(
        runtime, "_mapped_paths", lambda: (str(root / "torch/lib/cublas64_13.dll"),)
    )
    with pytest.raises(RuntimeError, match="import-order conflict.*separate processes"):
        runtime.load_libraries(root / "phydrax_cuda13_pjrt/lib/pjrt.dll")


def test_unresolved_import_is_evidence_not_empty_success(
    runtime: ModuleType, installed: tuple[Path, dict[str, metadata.Distribution]]
) -> None:
    root, distributions = installed
    root_dll = write_pe(
        root / "phydrax_cuda13_pjrt/lib/unknown.dll", delay_imports=("unowned.dll",)
    )
    distributions["jaxlib"] = _record(
        root, "jaxlib", "0.12.0", [root_dll.relative_to(root).as_posix()]
    )
    closure = runtime.dependency_closure(root_dll)
    assert not closure.complete
    assert closure.edges[0].kind == "unresolved" and closure.edges[0].path is None
    assert "unowned.dll" in closure.issues[0]
    with pytest.raises(RuntimeError, match="Cannot preload"):
        runtime.load_libraries(root_dll)


def test_unowned_root_cannot_be_loaded(
    runtime: ModuleType, installed: tuple[Path, dict[str, metadata.Distribution]]
) -> None:
    root, _ = installed
    unowned = write_pe(root / "cwd/decoy.dll")
    with pytest.raises(RuntimeError, match="RECORD owner"):
        runtime.load_libraries(unowned)


@pytest.fixture
def plugin(runtime: ModuleType, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module_path = runtime.__file__
    if module_path is None:
        raise RuntimeError("The runtime fixture has no source path")
    spec = importlib.util.spec_from_file_location(
        f"{runtime.__package__}.plugin", Path(module_path).with_name("plugin.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_wddm_default_and_explicit_allocator_options(
    plugin: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
    ):
        monkeypatch.delenv(name, raising=False)
    options = plugin._options()
    assert options["allocator"] == "default" and options["preallocate"] is False
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    assert plugin._options()["preallocate"] is True
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "bfc")
    monkeypatch.setenv("XLA_CLIENT_MEM_FRACTION", "0.25")
    explicit = plugin._options()
    assert explicit["allocator"] == "bfc" and explicit["memory_fraction"] == 0.25
    assert "preallocate" not in explicit


def test_invalid_user_allocator_is_not_masked(
    plugin: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "invalid-allocator")
    with pytest.raises(ValueError, match="ALLOCATOR"):
        plugin._options()


def test_unavailable_module_inventory_is_not_clean_evidence(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = OSError("EnumProcessModules access refused")

    def unavailable() -> tuple[str, ...]:
        raise failure

    monkeypatch.setattr(runtime, "_mapped_paths", unavailable)
    with pytest.raises(OSError) as result:
        runtime.mapped_libraries()
    assert result.value is failure


def test_unmatched_foreign_cudnn_family_is_not_adopted(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = installed
    monkeypatch.setattr(
        runtime, "_mapped_paths", lambda: (str(root / "torch/lib/cudnn_graph64_9.dll"),)
    )
    with pytest.raises(
        RuntimeError, match="Unowned resident CUDA DLL family.*separate processes"
    ):
        runtime.load_libraries(root / "phydrax_cuda13_pjrt/lib/pjrt.dll")


def test_record_path_escape_refuses(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
) -> None:
    root, distributions = installed
    write_pe(root.parent / "escape.dll")
    distributions["jaxlib"] = _record(root, "jaxlib", "0.12.0", ["../escape.dll"])
    with pytest.raises(RuntimeError, match="escapes"):
        runtime.dependency_closure(root / "phydrax_cuda13_pjrt/lib/pjrt.dll")


def test_mismatched_vendor_distribution_refuses_before_loading(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
) -> None:
    root, distributions = installed
    distributions["nvidia-cublas"] = _record(
        root, "nvidia-cublas", "12.9.0", ["nvidia/cu13/bin/x86_64/cublas64_13.dll"]
    )
    with pytest.raises(RuntimeError, match="13.4.92.*12.9.0"):
        runtime.installation_record()


def test_replaced_hardware_hook_is_refused(
    plugin: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jax._src import hardware_utils

    monkeypatch.setattr(hardware_utils, "has_visible_nvidia_gpu", lambda: False)
    with pytest.raises(RuntimeError, match="replaced JAX integration hook"):
        plugin._jax_contract.validate_loaded_hooks(hardware_utils)


def test_wrapped_backend_hook_is_refused(
    plugin: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jax._src import xla_bridge

    @functools.wraps(xla_bridge.backends)
    def replacement() -> NoReturn:
        raise AssertionError("replacement hook must never run")

    monkeypatch.setattr(xla_bridge, "backends", replacement)
    with pytest.raises(RuntimeError, match="replaced JAX integration hook"):
        plugin._jax_contract.validate_loaded_hooks(xla_bridge)


def test_duplicate_extension_roots_are_resolved_by_full_path(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, distributions = installed
    paths = [
        write_pe(root / "jaxlib/cpu/_sparse.pyd"),
        write_pe(root / "jaxlib/cuda/_sparse.pyd"),
    ]
    distributions["jaxlib"] = _record(
        root, "jaxlib", "0.12.0", [path.relative_to(root).as_posix() for path in paths]
    )
    monkeypatch.setattr(
        runtime,
        "_mapped_paths",
        lambda: (
            tuple(map(str, paths))
            + (
                str(root / "sklearn/tree/_utils.cp313-win_amd64.pyd"),
                str(root / "sklearn/manifold/_utils.cp313-win_amd64.pyd"),
                str(root / "unrelated/first/helper.dll"),
                str(root / "unrelated/second/helper.dll"),
            )
        ),
    )
    for path in paths:
        closure = runtime.dependency_closure(path)
        assert closure.complete and closure.issues == ()
        assert tuple(item.path for item in closure.objects) == (str(path.resolve()),)


@pytest.mark.parametrize("suffix", [".dll", ".pyd", ".exe"])
def test_import_basename_ambiguity_matters_when_required(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    suffix: str,
) -> None:
    root, distributions = installed
    paths = [
        write_pe(root / f"jaxlib/first/helper{suffix}"),
        write_pe(root / f"jaxlib/second/helper{suffix}"),
        write_pe(root / "jaxlib/root.pyd", imports=(f"helper{suffix}",)),
    ]
    distributions["jaxlib"] = _record(
        root, "jaxlib", "0.12.0", [path.relative_to(root).as_posix() for path in paths]
    )
    assert runtime.dependency_closure(paths[0]).complete
    with pytest.raises(RuntimeError, match="Ambiguous.*basename"):
        runtime.dependency_closure(paths[2])


def test_duplicate_resident_cuda_family_still_refuses(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = installed
    monkeypatch.setattr(
        runtime,
        "_mapped_paths",
        lambda: (
            str(root / "nvidia/cu13/bin/x86_64/cublas64_13.dll"),
            str(root / "torch/lib/cublas64_13.dll"),
        ),
    )
    with pytest.raises(RuntimeError, match="resident|import-order"):
        runtime.dependency_closure(root / "phydrax_cuda13_pjrt/lib/pjrt.dll")


@pytest.mark.parametrize("suffix", [".pyd", ".exe"])
def test_owned_non_dll_import_is_resolved(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    suffix: str,
) -> None:
    root, distributions = installed
    dependency = write_pe(root / f"jaxlib/helper{suffix}")
    requester = write_pe(root / "jaxlib/requester.pyd", imports=(f"helper{suffix}",))
    distributions["jaxlib"] = _record(
        root,
        "jaxlib",
        "0.12.0",
        [path.relative_to(root).as_posix() for path in (dependency, requester)],
    )
    closure = runtime.dependency_closure(requester)
    assert closure.complete
    assert closure.edges[0].path == str(dependency.resolve())


def test_driver_inspection_ignores_unrelated_resident_duplicates(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = installed
    monkeypatch.setattr(
        runtime,
        "_mapped_paths",
        lambda: (
            str(root / "jaxlib/cpu/_sparse.pyd"),
            str(root / "jaxlib/cuda/_sparse.pyd"),
            str(root / "first/helper.dll"),
            str(root / "second/helper.dll"),
        ),
    )
    failure = OSError("driver load deliberately refused")

    def refused(path: str, **kwargs: int | bool) -> NoReturn:
        assert Path(path).name == "nvcuda.dll"
        raise failure

    monkeypatch.setattr(runtime.ctypes, "WinDLL", refused, raising=False)
    with pytest.raises(OSError) as caught:
        runtime.driver_version()
    assert caught.value is failure


def test_required_resident_non_dll_ambiguity_refuses(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, distributions = installed
    dependency = write_pe(root / "jaxlib/helper.exe")
    requester = write_pe(root / "jaxlib/requester.pyd", imports=("helper.exe",))
    distributions["jaxlib"] = _record(
        root,
        "jaxlib",
        "0.12.0",
        [path.relative_to(root).as_posix() for path in (dependency, requester)],
    )
    monkeypatch.setattr(
        runtime,
        "_mapped_paths",
        lambda: (
            str(dependency),
            str(root / "foreign/helper.exe"),
        ),
    )
    with pytest.raises(RuntimeError, match="Multiple resident.*required basename"):
        runtime.dependency_closure(requester)


def _driver_files(
    root: Path, runtime: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    system = root / "Windows/System32"
    package = system / "DriverStore/FileRepository/nv_fixture"
    package.mkdir(parents=True)
    paths = (
        package / "nvcudart_hybrid64.dll",
        package / "nvcuda64.dll",
        system / "nvcuda.dll",
    )
    for path in paths:
        path.write_bytes(path.name.encode("ascii"))
    monkeypatch.setattr(runtime, "_mapped_paths", lambda: tuple(map(str, paths)))
    return paths


@pytest.mark.parametrize(
    "state",
    ["dispatch", "backing", "both", "foreign", "other-cohort", "unobserved-cohort"],
)
def test_nvml_dispatch_admits_only_the_observed_driver_cohort(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    root, distributions = installed
    driver_paths: tuple[Path, ...] = _driver_files(root, runtime, monkeypatch)
    system = driver_paths[2].parent
    dispatch = write_pe(system / "nvml.dll")
    backing = write_pe(driver_paths[1].parent / "nvml.dll")
    mapped = (
        [dispatch]
        if state == "dispatch"
        else [backing]
        if state == "backing"
        else [dispatch, backing]
    )
    if state == "foreign":
        mapped.append(write_pe(root / "unowned/nvml.dll"))
    elif state == "other-cohort":
        mapped.append(write_pe(system / "DriverStore/FileRepository/other/nvml.dll"))
    elif state == "unobserved-cohort":
        driver_paths = (driver_paths[0], driver_paths[2])
    monkeypatch.setattr(
        runtime,
        "_mapped_paths",
        lambda: tuple(str(path) for path in (*driver_paths, *mapped)),
    )
    companion = write_pe(
        root / "phydrax_cuda13/lib/phydrax_cuda_quda.dll", imports=("nvml.dll",)
    )
    distributions["phydrax-cuda13"] = _record(
        root, "phydrax-cuda13", "0.2.2", [companion.relative_to(root).as_posix()]
    )
    if state in ("foreign", "other-cohort", "unobserved-cohort"):
        with pytest.raises(RuntimeError, match="unexpected driver owner"):
            runtime.dependency_closure(companion)
    else:
        closure = runtime.dependency_closure(companion)
        assert closure.complete and closure.issues == ()
        assert closure.edges[0].owner == "NVIDIA driver"
        assert Path(closure.edges[0].path) == dispatch


def test_loaded_nvml_dispatch_still_refuses_a_later_foreign_owner(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, distributions = installed
    driver_paths = _driver_files(root, runtime, monkeypatch)
    dispatch = write_pe(driver_paths[2].parent / "nvml.dll")
    backing = write_pe(driver_paths[1].parent / "nvml.dll")
    mapped = [*driver_paths, dispatch, backing]
    monkeypatch.setattr(runtime, "_mapped_paths", lambda: tuple(map(str, mapped)))
    monkeypatch.setattr(
        runtime.os,
        "add_dll_directory",
        lambda _: SimpleNamespace(close=lambda: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime.ctypes,
        "WinDLL",
        lambda _path, *, winmode: ctypes.CDLL(None, handle=0),
        raising=False,
    )
    companion = write_pe(
        root / "phydrax_cuda13/lib/phydrax_cuda_quda.dll", imports=("nvml.dll",)
    )
    distributions["phydrax-cuda13"] = _record(
        root, "phydrax-cuda13", "0.2.2", [companion.relative_to(root).as_posix()]
    )
    runtime.load_libraries(companion)
    foreign = write_pe(root / "unowned/nvml.dll")
    mapped.append(foreign)
    with pytest.raises(RuntimeError, match="unexpected driver owner") as caught:
        runtime.load_libraries(companion)
    assert str(foreign) in str(caught.value)


def test_runtime_identity_observes_driver_cohort_without_cuda_calls(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = installed
    paths = _driver_files(root, runtime, monkeypatch)

    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("Passive identity must not load or initialize CUDA")

    monkeypatch.setattr(runtime.ctypes, "WinDLL", forbidden, raising=False)
    monkeypatch.setattr(runtime, "cuda_runtime_api", forbidden)
    identity = runtime.runtime_identity()
    assert identity.linkage == "driver-hybrid"
    assert identity.owner == "display-driver"
    for prefix, path in zip(("runtime", "driver", "driver_dispatch"), paths):
        assert getattr(identity, prefix + "_path") == str(path.resolve())
        assert (
            getattr(identity, prefix + "_sha256")
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )
    assert "cuda_major" not in identity._asdict()
    assert "runtime_version" not in identity._asdict()
    assert json.loads(json.dumps(identity._asdict())) == identity._asdict()


@pytest.mark.parametrize("missing", (0, 1, 2))
def test_runtime_identity_requires_observed_hybrid_and_actual_driver(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
    missing: int,
) -> None:
    root, _ = installed
    paths = _driver_files(root, runtime, monkeypatch)
    monkeypatch.setattr(
        runtime,
        "_mapped_paths",
        lambda: tuple(str(path) for index, path in enumerate(paths) if index != missing),
    )
    with pytest.raises(RuntimeError, match="Missing resident.*" + paths[missing].name):
        runtime.runtime_identity()


@pytest.mark.parametrize(
    "conflict",
    ("hybrid-owner", "driver-owner", "dispatch-owner", "cohort", "duplicate", "shared"),
)
def test_runtime_identity_and_admission_refuse_conflicting_driver_cohorts(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
    conflict: str,
) -> None:
    root, _ = installed
    paths = list(_driver_files(root, runtime, monkeypatch))
    match conflict:
        case "hybrid-owner":
            paths[0] = root / "foreign/nvcudart_hybrid64.dll"
        case "driver-owner":
            paths[1] = root / "foreign/nvcuda64.dll"
        case "dispatch-owner":
            paths[2] = root / "foreign/nvcuda.dll"
        case "cohort":
            paths[1] = paths[1].parent.parent / "other_package/nvcuda64.dll"
        case "duplicate":
            paths.append(root / "foreign/nvcudart_hybrid64.dll")
        case "shared":
            paths.append(root / "foreign/cudart64_13.dll")
    monkeypatch.setattr(runtime, "_mapped_paths", lambda: tuple(map(str, paths)))
    for operation in (
        runtime.runtime_identity,
        lambda: runtime.dependency_closure(root / "phydrax_cuda13_pjrt/lib/pjrt.dll"),
    ):
        with pytest.raises(
            RuntimeError,
            match="unexpected owner|cohort|Multiple resident|Conflicting resident",
        ):
            operation()


def _runtime_accessor(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing: str | None = None,
    linkage: str = "driver-hybrid",
) -> tuple[SimpleNamespace, list[str], list[str]]:
    root, distributions = installed
    path = write_pe(root / "phydrax_cuda13/lib/phydrax_cuda_cudss.dll")
    manifest = root / "phydrax_cuda13/phx_cuda_build.json"
    manifest.write_text(
        json.dumps(
            {
                "runtime_linkage": {
                    "kind": linkage,
                    "target_dll": "nvcudart_hybrid64.dll",
                    "owner": "display-driver",
                }
            }
        ),
        encoding="utf-8",
    )
    distributions["phydrax-cuda13"] = _record(
        root,
        "phydrax-cuda13",
        "0.2.2",
        [item.relative_to(root).as_posix() for item in (path, manifest)],
    )
    convention = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
    monkeypatch.setattr(runtime.ctypes, "WINFUNCTYPE", convention, raising=False)
    calls: list[str] = []
    lookups: list[str] = []
    pointers: dict[str, Any] = {}
    for name, arguments in runtime._CUDA_SIGNATURES.items():

        def invoke(*args: Any, symbol: str = name) -> int:
            calls.append(symbol)
            if symbol == "cudaRuntimeGetVersion":
                args[0][0] = 14010
            return 0

        pointers[name] = convention(ctypes.c_int, *arguments)(invoke)

    def accessor(raw: bytes) -> int | None:
        name = raw.decode("ascii")
        lookups.append(name)
        return (
            None
            if name == missing
            else ctypes.cast(pointers[name], ctypes.c_void_p).value
        )

    companion = SimpleNamespace(
        phx_cuda_runtime_function=accessor,
        phx_cuda_last_error=lambda: b"unknown runtime function",
        callbacks=pointers,
    )
    runtime._LIBRARIES[runtime._key(path)] = companion
    return companion, calls, lookups


def test_canonical_runtime_binding_retains_owner_and_uses_declared_abi(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    companion, calls, lookups = _runtime_accessor(runtime, installed, monkeypatch)
    api = runtime.cuda_runtime_api(companion)
    assert api._library is companion
    assert calls == []  # Pointer lookup is not runtime/device initialization.
    assert set(lookups) == set(runtime._CUDA_SIGNATURES)
    assert runtime.cuda_runtime_api(companion) is api
    version = ctypes.c_int()
    assert api.cudaRuntimeGetVersion(ctypes.byref(version)) == 0
    assert version.value == 14010  # Compatible driver level, not a wheel major.
    assert calls == ["cudaRuntimeGetVersion"]
    for unknown in ("cudaLaunchKernel", "", None):
        with pytest.raises(ValueError, match="Unknown CUDA runtime API"):
            api.function(unknown)
    with pytest.raises(AttributeError):
        getattr(api, "cudaLaunchKernel")
    assert len(lookups) == len(runtime._CUDA_SIGNATURES)


def test_runtime_accessor_failure_preserves_native_error_and_is_not_cached(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    companion, calls, _ = _runtime_accessor(
        runtime, installed, monkeypatch, missing="cudaMemGetInfo"
    )
    with pytest.raises(RuntimeError, match="cudaMemGetInfo: unknown runtime function"):
        runtime.cuda_runtime_api(companion)
    assert runtime._CUDA_APIS == {}
    assert calls == []
    with pytest.raises(RuntimeError, match="already admitted"):
        runtime.cuda_runtime_api(object())


def test_driver_compatibility_version_is_not_a_wheel_major(
    runtime: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def version(output: Any) -> int:
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = 14010
        return 0

    monkeypatch.setattr(
        runtime,
        "cuda_driver_library",
        lambda: SimpleNamespace(cuDriverGetVersion=version),
    )
    assert runtime.driver_version() == 14010


def test_runtime_api_refuses_non_hybrid_companion_linkage(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    companion, calls, lookups = _runtime_accessor(
        runtime, installed, monkeypatch, linkage="static"
    )
    with pytest.raises(RuntimeError, match="canonical driver-hybrid runtime linkage"):
        runtime.cuda_runtime_api(companion)
    assert calls == lookups == []


def test_runtime_api_refuses_missing_companion_accessor(
    runtime: ModuleType,
    installed: tuple[Path, dict[str, metadata.Distribution]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    companion, calls, lookups = _runtime_accessor(runtime, installed, monkeypatch)
    del companion.phx_cuda_runtime_function
    with pytest.raises(RuntimeError, match="lacks the public runtime accessor"):
        runtime.cuda_runtime_api(companion)
    assert calls == lookups == []
