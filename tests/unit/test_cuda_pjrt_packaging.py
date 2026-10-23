"""Exercise wheel contents using staged PE fixtures, not mocked PE inspection."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import shutil
import struct
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
NATIVE = REPO / "native" / "cuda13_pjrt"


def _pe(export: str, machine: int = 0x8664) -> bytes:
    """Small well-formed PE32+ image with one direct named export."""
    data = bytearray(0x600)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", data, 0x84, machine, 1, 0, 0, 0, 0xF0, 0x2022)
    optional = 0x98
    struct.pack_into("<H", data, optional, 0x20B)
    struct.pack_into("<II", data, optional + 32, 0x1000, 0x200)
    struct.pack_into("<II", data, optional + 56, 0x2000, 0x200)
    struct.pack_into("<I", data, optional + 108, 16)
    struct.pack_into("<II", data, optional + 112, 0x1000, 0x80)
    section = optional + 0xF0
    data[section : section + 8] = b".rdata\0\0"
    struct.pack_into("<IIII", data, section + 8, 0x400, 0x1000, 0x400, 0x200)
    struct.pack_into("<I", data, section + 36, 0x40000040)
    struct.pack_into(
        "<IIHHIIIIIII", data, 0x200, 0, 0, 0, 0, 0x10A0, 1, 1, 1, 0x1098, 0x1080, 0x1090
    )
    struct.pack_into("<I", data, 0x280, 0x10C0)
    struct.pack_into("<H", data, 0x290, 0)
    struct.pack_into("<I", data, 0x298, 0x1100)
    data[0x2A0:0x2A8] = b"test.dll"
    encoded = export.encode() + b"\0"
    data[0x2C0 : 0x2C0 + len(encoded)] = encoded
    return bytes(data)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _seal(stage: Path, record: dict[str, Any]) -> None:
    identity = {key: value for key, value in record.items() if key != "build_id"}
    record["build_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write(stage / "phydrax_cuda13_pjrt/manifest.json", json.dumps(record).encode())


def _settings(stage: Path) -> dict[str, str]:
    return {
        "staging": str(stage),
        "kernel-wheel": str(
            stage.parent / "jax_cuda13_plugin-0.12.0-cp313-cp313-win_amd64.whl"
        ),
    }


@pytest.fixture
def staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ModuleType, Path, dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(
        "cuda_pjrt_packaging", NATIVE / "_build_backend.py"
    )
    assert spec is not None and spec.loader is not None
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    root = tmp_path / "repo/native/cuda13_pjrt"
    root.mkdir(parents=True)
    shutil.copyfile(NATIVE / "pyproject.toml", root / "pyproject.toml")
    shutil.copyfile(REPO / "pyproject.toml", root.parents[1] / "pyproject.toml")
    shutil.copyfile(REPO / "LICENSE", root.parents[1] / "LICENSE")
    _write(
        root.parents[1] / "phydrax/_pe_dependencies.py",
        (REPO / "phydrax/_pe_dependencies.py").read_bytes(),
    )
    monkeypatch.setattr(backend, "ROOT", root)
    monkeypatch.setattr(backend, "_host", lambda: None)
    stage = tmp_path / "stage"
    sources = {
        "phydrax_cuda13_pjrt/__init__.py": b'"""Packaging fixture."""\n',
        "phydrax_cuda13_pjrt/plugin.py": b'def initialize() -> None:\n    raise RuntimeError("fixture")\n',
        "jax_cuda13_plugin/__init__.py": b'"""Kernel package fixture."""\n',
    }
    assets: list[dict[str, Any]] = []
    for name, data in sources.items():
        _write(root / "python" / name, data)
    sources["phydrax_cuda13_pjrt/_pe_dependencies.py"] = (
        REPO / "phydrax/_pe_dependencies.py"
    ).read_bytes()
    sources["phydrax_cuda13_pjrt/lib/pjrt.dll"] = _pe("GetPjrtApi")
    sources["jax_cuda13_plugin/_solver.cp313-win_amd64.pyd"] = _pe("PyInit__solver")
    sources["jax_cuda13_plugin/version.py"] = b'__version__ = "0.12.0"\n'
    kernel_wheel = Path(_settings(stage)["kernel-wheel"])
    with zipfile.ZipFile(kernel_wheel, "w") as wheel:
        for name, data in sources.items():
            if name.startswith("jax_cuda13_plugin/"):
                wheel.writestr(name, data)
        wheel.writestr(
            "jax_cuda13_plugin-0.12.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: jax-cuda13-plugin\nVersion: 0.12.0\n",
        )
        wheel.writestr(
            "jax_cuda13_plugin-0.12.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nTag: cp313-cp313-win_amd64\n",
        )
    for name, data in sorted(sources.items()):
        _write(stage / name, data)
        role = (
            "pjrt"
            if name.endswith(".dll")
            else "kernel"
            if name.endswith(".pyd")
            else "parser"
            if name.endswith("_pe_dependencies.py")
            else "runtime"
        )
        assets.append(
            {
                "path": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
                "role": role,
            }
        )
    license_data = (REPO / "LICENSE").read_bytes()
    license_path = "licenses/Phydrax-LICENSE"
    _write(stage / license_path, license_data)
    licenses = [
        {
            "path": license_path,
            "sha256": hashlib.sha256(license_data).hexdigest(),
            "size": len(license_data),
        }
    ]
    lock = {
        "sources": [
            {"name": "jax", "revision": "8dcb7aeced770bd22531b42d0b925c7342787c27"},
            {"name": "xla", "revision": "d867969800d4a8fd045505eabc65e1e66f30794f"},
        ],
        "patches": [],
        "toolchain": {"bazel_version": "9.2.0"},
        "licenses": {license_path: licenses[0]["sha256"]},
        "required_kernels": ["_solver"],
    }
    recipe_data = b"# Synthetic build recipe for packaging tests.\n"
    _write(root.parents[1] / "tools/build_cuda_pjrt_wheels.py", recipe_data)
    lock["recipe_inputs"] = {
        "tools/build_cuda_pjrt_wheels.py": hashlib.sha256(recipe_data).hexdigest()
    }
    lock_bytes = json.dumps(lock).encode()
    (root / "sources.lock.json").write_bytes(lock_bytes)
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    record = {
        "distribution": "phydrax-cuda13-pjrt",
        "version": version,
        "cuda_major": 13,
        "jax_version": "0.12.0",
        "jaxlib_version": "0.12.0",
        "runtime_requirements": tomllib.loads((root / "pyproject.toml").read_text())[
            "project"
        ]["dependencies"],
        "python_tag": "cp313",
        "abi_tag": "cp313",
        "platform_tag": "win_amd64",
        "sources_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "sources": lock["sources"],
        "patches": [],
        "toolchain": lock["toolchain"],
        "capabilities": [],
        "assets": assets,
        "licenses": licenses,
    }
    record["kernel_wheel"] = {
        "filename": kernel_wheel.name,
        "sha256": hashlib.sha256(kernel_wheel.read_bytes()).hexdigest(),
    }
    _seal(stage, record)
    return backend, stage, record


def test_staged_wheel_metadata_assets_record_and_licenses(
    staged: tuple[ModuleType, Path, dict[str, Any]], tmp_path: Path
) -> None:
    backend, stage, record = staged
    output = tmp_path / "wheels"
    filename = backend.build_wheel(str(output), _settings(stage))
    first = (output / filename).read_bytes()
    assert filename.endswith("-cp313-cp313-win_amd64.whl")
    backend.build_wheel(str(output), _settings(stage))
    assert (output / filename).read_bytes() == first
    with zipfile.ZipFile(io.BytesIO(first)) as wheel:
        names = wheel.namelist()
        info = next(
            name.rsplit("/", 1)[0] for name in names if name.endswith("/METADATA")
        )
        metadata = BytesParser().parsebytes(wheel.read(f"{info}/METADATA"))
        assert metadata["Name"] == "phydrax-cuda13-pjrt"
        assert metadata["Version"] == record["version"]
        dependencies = metadata.get_all("Requires-Dist")
        assert dependencies is not None
        assert "jax==0.12.0" in dependencies and "jaxlib==0.12.0" in dependencies
        assert not any(
            any(word in dependency for word in ("nccl", "nvshmem", "nvcc", "winjax"))
            for dependency in dependencies
        )
        assert metadata.get_all("License-File") == ["Phydrax-LICENSE"]
        assert (
            wheel.read(f"{info}/licenses/Phydrax-LICENSE")
            == (REPO / "LICENSE").read_bytes()
        )
        assert (
            wheel.read(f"{info}/entry_points.txt")
            == b"[jax_plugins]\ncuda = phydrax_cuda13_pjrt.plugin\n"
        )
        assert b"Tag: cp313-cp313-win_amd64" in wheel.read(f"{info}/WHEEL")
        for asset in record["assets"]:
            assert wheel.read(asset["path"]) == (stage / asset["path"]).read_bytes()
        rows = list(csv.reader(io.StringIO(wheel.read(f"{info}/RECORD").decode())))
        assert {row[0] for row in rows} == set(names)
        for name, digest, size in rows:
            if name.endswith("/RECORD"):
                assert (digest, size) == ("", "")
            else:
                data = wheel.read(name)
                assert int(size) == len(data)
                assert (
                    digest
                    == "sha256="
                    + base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                    .rstrip(b"=")
                    .decode()
                )


@pytest.mark.parametrize(
    "change",
    [
        "malformed",
        "hash",
        "missing",
        "tag",
        "foreign",
        "escape",
        "duplicate",
        "license",
        "runtime",
        "parser",
        "export",
        "machine",
        "unlisted",
        "kernel_missing",
        "kernel_conflict",
        "source",
        "runtime_requirements",
    ],
)
def test_refuses_invalid_staging(
    staged: tuple[ModuleType, Path, dict[str, Any]], tmp_path: Path, change: str
) -> None:
    backend, stage, record = staged
    pjrt = next(asset for asset in record["assets"] if asset["role"] == "pjrt")
    path = stage / pjrt["path"]
    if change == "malformed":
        (stage / backend.MANIFEST).write_bytes(b'{"assets":')
    elif change == "missing":
        path.unlink()
    elif change == "hash":
        path.write_bytes(path.read_bytes() + b"tampered")
    else:
        if change == "tag":
            record["abi_tag"] = "cp313t"
        elif change == "source":
            record["sources"] = []
        elif change == "runtime_requirements":
            record["runtime_requirements"] = ["jax==0.12.0"]
        elif change == "kernel_missing":
            kernel = next(item for item in record["assets"] if item["role"] == "kernel")
            (stage / kernel["path"]).unlink()
            record["assets"].remove(kernel)
        elif change == "kernel_conflict":
            kernel = dict(
                next(item for item in record["assets"] if item["role"] == "kernel")
            )
            data = (stage / kernel["path"]).read_bytes()
            kernel["path"] = "jax_cuda13_plugin/_solver.pyd"
            _write(stage / kernel["path"], data)
            record["assets"].append(kernel)
        elif change == "foreign":
            pjrt["path"] = "jaxlib/pjrt.dll"
        elif change == "escape":
            pjrt["path"] = "../pjrt.dll"
        elif change == "duplicate":
            record["assets"].append(dict(pjrt))
        elif change == "license":
            record["licenses"] = []
        elif change == "unlisted":
            _write(stage / "jax_plugins/cuda/__init__.py", b"")
        else:
            asset = pjrt
            if change in ("runtime", "parser"):
                suffix = "plugin.py" if change == "runtime" else "_pe_dependencies.py"
                asset = next(
                    item for item in record["assets"] if item["path"].endswith(suffix)
                )
                data = b"# altered runtime\n"
            else:
                data = (
                    _pe("WrongExport") if change == "export" else _pe("GetPjrtApi", 0x14C)
                )
            _write(stage / asset["path"], data)
            asset.update(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
        _seal(stage, record)
    with pytest.raises(backend.StagingError):
        backend.build_wheel(str(tmp_path / "wheel"), _settings(stage))


def test_source_build_and_editable_refusal(
    staged: tuple[ModuleType, Path, dict[str, Any]], tmp_path: Path
) -> None:
    backend, _, _ = staged
    with pytest.raises(backend.StagingError, match="staging"):
        backend.build_wheel(str(tmp_path))
    for hook in (
        backend.build_sdist,
        backend.build_editable,
        backend.prepare_metadata_for_build_editable,
    ):
        with pytest.raises(backend.StagingError):
            hook(str(tmp_path))
    for hook in (
        backend.get_requires_for_build_sdist,
        backend.get_requires_for_build_editable,
    ):
        with pytest.raises(backend.StagingError):
            hook()


def test_prepared_metadata_is_checked(
    staged: tuple[ModuleType, Path, dict[str, Any]], tmp_path: Path
) -> None:
    backend, stage, _ = staged
    name = backend.prepare_metadata_for_build_wheel(str(tmp_path / "metadata"))
    prepared = tmp_path / "metadata" / name
    (prepared / "METADATA").write_text("changed", encoding="utf-8")
    with pytest.raises(backend.StagingError, match="Prepared metadata"):
        backend.build_wheel(str(tmp_path / "wheel"), _settings(stage), str(prepared))


@pytest.mark.parametrize(
    "platform_name,machine,python_version,free_threaded",
    [
        ("linux", "AMD64", (3, 13), 0),
        ("win32", "ARM64", (3, 13), 0),
        ("win32", "AMD64", (3, 12), 0),
        ("win32", "AMD64", (3, 13), 1),
    ],
)
def test_refuses_incompatible_packaging_host(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    machine: str,
    python_version: tuple[int, int],
    free_threaded: int,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "cuda_pjrt_host_packaging", NATIVE / "_build_backend.py"
    )
    assert spec is not None and spec.loader is not None
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    monkeypatch.setattr(backend.sys, "platform", platform_name)
    monkeypatch.setattr(backend.platform, "machine", lambda: machine)
    monkeypatch.setattr(backend.sys, "version_info", python_version)
    monkeypatch.setattr(backend.sysconfig, "get_config_var", lambda name: free_threaded)
    with pytest.raises(backend.StagingError, match="CPython"):
        backend._host()


def test_kernel_wheel_provenance_is_required(
    staged: tuple[ModuleType, Path, dict[str, Any]], tmp_path: Path
) -> None:
    backend, stage, record = staged
    with pytest.raises(backend.StagingError, match="kernel-wheel"):
        backend.build_wheel(str(tmp_path / "wheel"), {"staging": str(stage)})
    version = next(
        asset
        for asset in record["assets"]
        if asset["path"] == "jax_cuda13_plugin/version.py"
    )
    data = b'__version__ = "0.11.2"\n'
    _write(stage / version["path"], data)
    version.update(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
    _seal(stage, record)
    with pytest.raises(backend.StagingError, match="Kernel wheel member"):
        backend.build_wheel(str(tmp_path / "wheel"), _settings(stage))


def test_rejects_changed_locked_recipe(
    staged: tuple[ModuleType, Path, dict[str, Any]], tmp_path: Path
) -> None:
    backend, stage, _ = staged
    (backend.ROOT.parents[1] / "tools/build_cuda_pjrt_wheels.py").write_bytes(
        b"# changed\n"
    )
    with pytest.raises(backend.StagingError, match="recipe input digest"):
        backend.build_wheel(str(tmp_path / "wheel"), _settings(stage))


@pytest.mark.parametrize("locked,solver", [(True, True), (False, True), (True, False)])
def test_sparse_qr_requires_locked_source_and_solver(
    staged: tuple[ModuleType, Path, dict[str, Any]],
    tmp_path: Path,
    locked: bool,
    solver: bool,
) -> None:
    backend, stage, record = staged
    lock_path = backend.ROOT / "sources.lock.json"
    lock = json.loads(lock_path.read_bytes())
    lock["capabilities"] = ["sparse_qr"] if locked else []
    if not solver:
        lock["required_kernels"] = []
    lock_bytes = json.dumps(lock).encode()
    lock_path.write_bytes(lock_bytes)
    record["sources_lock_sha256"] = hashlib.sha256(lock_bytes).hexdigest()
    record["capabilities"] = ["sparse_qr"]
    _seal(stage, record)
    if locked and solver:
        filename = backend.build_wheel(str(tmp_path / "wheel"), _settings(stage))
        assert (tmp_path / "wheel" / filename).is_file()
    else:
        with pytest.raises(backend.StagingError, match="sparse_qr"):
            backend.build_wheel(str(tmp_path / "wheel"), _settings(stage))
