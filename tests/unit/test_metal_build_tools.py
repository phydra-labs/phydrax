from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tools import build_metal_wheels as build


@pytest.fixture
def native_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    lock = build.read_lock()
    identity, files = build.source_identity(lock)
    original = build.ROOT
    for relative in files:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / relative, target)
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "NATIVE", tmp_path / "native/metal")
    monkeypatch.setattr(build, "LOCK", tmp_path / "native/metal/sources.lock.json")
    monkeypatch.setattr(build, "__file__", str(tmp_path / "tools/build_metal_wheels.py"))
    return tmp_path, identity


def test_native_source_identity_is_independent_of_checkout_location(
    native_tree: tuple[Path, str],
) -> None:
    _, expected = native_tree
    observed, _ = build.source_identity(build.read_lock())
    assert observed == expected


@pytest.mark.parametrize(
    "relative",
    [
        "native/metal/python/phydrax_metal/plugin.py",
        "native/metal/_build_backend.py",
        "native/metal/CMakeLists.txt",
        "tools/build_metal_wheels.py",
    ],
    ids=["runtime-loader", "wheel-backend", "native-build", "build-driver"],
)
def test_material_build_inputs_change_native_source_identity(
    native_tree: tuple[Path, str], relative: str
) -> None:
    root, previous = native_tree
    source = root / relative
    source.write_bytes(source.read_bytes() + b"\n# changed source input\n")
    observed, _ = build.source_identity(build.read_lock())
    assert observed != previous


def test_source_lock_changes_after_preparation_refuse(
    native_tree: tuple[Path, str],
) -> None:
    root, _ = native_tree
    lock = build.read_lock()
    path = root / "native/metal/sources.lock.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(build.BuildFailure):
        build.source_identity(lock)


def test_changed_native_patch_cannot_reuse_locked_identity(
    native_tree: tuple[Path, str],
) -> None:
    root, _ = native_tree
    lock = build.read_lock()
    patch = next(item for item in lock.patches if item.origin == "native/metal")
    path = root / "native/metal" / patch.path
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(build.BuildFailure):
        build.source_identity(lock)
