"""Host-only contracts for the RECORD-owned Windows companion toolkit."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load test subject: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def toolkit() -> ModuleType:
    return load_module(
        "_test_windows_cuda_toolkit", ROOT / "native/cuda/cmake/windows_toolkit.py"
    )


@pytest.mark.parametrize(
    ("record", "relative"),
    [
        ("nvidia/cu13/include/cuda.h", "include/cuda.h"),
        ("nvidia/cu13/lib/x64/cudart.lib", "lib/x64/cudart.lib"),
        ("nvidia/cu13/bin/nvcc.exe", "bin/nvcc.exe"),
        ("nvidia/cu13/nvvm/libdevice/libdevice.10.bc", "nvvm/libdevice/libdevice.10.bc"),
        ("nvidia/cu13/include/__init__.py", None),
    ],
)
def test_windows_toolkit_record_layout(
    toolkit: ModuleType, record: str, relative: str | None
) -> None:
    result = toolkit.record_relative(tuple(record.split("/")))
    assert result == (Path(relative) if relative is not None else None)


def test_windows_toolkit_copy_is_owned_and_detects_stale_bytes(
    toolkit: ModuleType, tmp_path: Path
) -> None:
    source = tmp_path / "vendor.dll"
    source.write_bytes(b"verified vendor bytes")
    destination = tmp_path / "view/bin/vendor.dll"
    sha256 = toolkit.digest(source)
    toolkit.copy_verified(source, destination, sha256)
    toolkit.copy_verified(source, destination, sha256)
    assert destination.read_bytes() == source.read_bytes()
    assert not destination.is_symlink()
    destination.write_bytes(b"unowned stale bytes")
    with pytest.raises(RuntimeError, match="Conflicting/stale"):
        toolkit.copy_verified(source, destination, sha256)
    assert source.read_bytes() == b"verified vendor bytes"


def test_windows_toolkit_refuses_tampered_wheel_record(
    toolkit: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "site"
    record = "nvidia/cu13/include/cuda.h"
    source = site / record
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original")
    info = site / "nvidia_cuda_runtime-13.4.92.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Name: nvidia-cuda-runtime\nVersion: 13.4.92\n", encoding="utf-8"
    )
    hashed = (
        base64.urlsafe_b64encode(hashlib.sha256(source.read_bytes()).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    (info / "RECORD").write_text(f"{record},sha256={hashed},8\n", encoding="utf-8")
    distribution = importlib.metadata.PathDistribution(info)
    monkeypatch.setattr(toolkit.metadata, "distributions", lambda: [distribution])
    inventory = toolkit.wheel_inventory(tmp_path / "view")
    assert inventory["wheels"] == {"nvidia-cuda-runtime": "13.4.92"}
    source.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="RECORD mismatch"):
        toolkit.wheel_inventory(tmp_path / "other-view")
    assert not (tmp_path / "other-view/include/cuda.h").exists()


def test_component_manifest_distinguishes_unshipped_and_missing_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = load_module(
        "_test_cuda13_assets", ROOT / "native/cuda13/python/phydrax_cuda13/__init__.py"
    )
    monkeypatch.setattr(package, "__file__", str(tmp_path / "__init__.py"))
    (tmp_path / "components.json").write_text(
        json.dumps({"cudss": "lib/phydrax_cuda_cudss.dll"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unavailable"):
        package.component_library("cutensornet")
    with pytest.raises(FileNotFoundError, match="library is missing"):
        package.component_library("cudss")
    dll = tmp_path / "lib/phydrax_cuda_cudss.dll"
    dll.parent.mkdir()
    dll.write_bytes(b"asset location fixture, not executable")
    assert package.component_library("cudss") == dll


@pytest.mark.parametrize(
    "relative", ["../outside.dll", "/absolute.dll", "lib/../outside.dll", 1]
)
def test_component_manifest_refuses_paths_outside_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str | int
) -> None:
    package = load_module(
        "_test_cuda13_bad_assets",
        ROOT / "native/cuda13/python/phydrax_cuda13/__init__.py",
    )
    monkeypatch.setattr(package, "__file__", str(tmp_path / "__init__.py"))
    (tmp_path / "components.json").write_text(
        json.dumps({"cudss": relative}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="Invalid CUDA component path"):
        package.component_library("cudss")
