from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
        "phydrax/_jax_indexing.py",
    ],
    ids=[
        "runtime-loader",
        "wheel-backend",
        "native-build",
        "build-driver",
        "canonical-indexing",
    ],
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


@pytest.fixture
def staged_wheel(
    native_tree: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Path]:
    """Exercise packaging with inert fixture assets, never native qualification."""
    root, source_id = native_tree
    spec = importlib.util.spec_from_file_location(
        "_metal_backend_under_test", root / "native/metal/_build_backend.py"
    )
    assert spec is not None and spec.loader is not None
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    monkeypatch.setattr(backend, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(backend.platform, "machine", lambda: "arm64")
    (root / "LICENSE").write_text("Unit-test packaging license fixture.\n")
    stage = root / "stage"
    package = stage / "phydrax_metal"
    (package / "lib").mkdir(parents=True)
    (package / "lib/libpjrt_plugin_mps.dylib").write_bytes(b"inert packaging fixture")
    (package / "lib/mlx.metallib").write_bytes(b"inert packaging fixture")
    for relative in backend._license_names():
        if relative == "Phydrax-LICENSE":
            continue
        target = stage / "licenses" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("Unit-test packaging license fixture.\n")
    trees = {"xla": root / "xla", "stablehlo": root / "stablehlo"}
    headers = {
        trees["xla"] / "xla/pjrt/c/pjrt_c_api.h": (
            "#define PJRT_API_MAJOR 0\n#define PJRT_API_MINOR 75\n"
        ),
        trees["stablehlo"] / "stablehlo/dialect/Version.h": (
            "getCurrentVersion() { return Version(1, 13, 0); }\n"
        ),
    }
    for path, content in headers.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    executable = Path(sys.executable)
    configuration = build.Configuration(
        python=executable,
        cmake=executable,
        ninja=executable,
        build_dir=root,
        wheelhouse=root / "wheelhouse",
        output=root / "packaging.json",
        download_dir=root / "downloads",
        jobs=1,
        dependencies_only=False,
    )
    tools = build.Toolchain(
        cc=executable,
        cxx=executable,
        sysroot=root,
        identity={"purpose": "inert unit-test packaging fixture"},
    )
    build.manifest(
        configuration,
        build.read_lock(),
        tools,
        trees,
        stage,
        {"minos": "14.0"},
        backend._project()[1],
        source_id,
    )
    return backend, stage


def test_companion_wheel_binds_and_imports_private_standalone_indexing(
    staged_wheel: tuple[Any, Path], tmp_path: Path
) -> None:
    backend, stage = staged_wheel
    wheel = tmp_path / backend.build_wheel(
        str(tmp_path), config_settings={"staging": str(stage)}
    )
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert all(not name.startswith("phydrax/") for name in names)
        helper = archive.read("phydrax_metal/_jax_indexing.py")
        assert helper == (build.ROOT / "phydrax/_jax_indexing.py").read_bytes()
        manifest = json.loads(archive.read("phydrax_metal/build.json"))
        assert manifest["assets"]["_jax_indexing.py"] == build.digest(helper)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from phydrax_metal import _jax_indexing, plugin; "
            "assert 'phydrax' not in sys.modules; "
            "assert 'jax' not in sys.modules; "
            "_jax_indexing.register_gather_lowering('cpu'); "
            "from jax._src import xla_bridge; "
            "assert not xla_bridge.backends_are_initialized(); "
            "assert 'phydrax' not in sys.modules",
            str(wheel),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_companion_packaging_refuses_changed_canonical_indexing(
    staged_wheel: tuple[Any, Path], tmp_path: Path
) -> None:
    backend, stage = staged_wheel
    source = build.ROOT / "phydrax/_jax_indexing.py"
    source.write_bytes(source.read_bytes() + b"\n# canonical input changed\n")
    with pytest.raises(backend.StagingError, match="canonical source"):
        backend.build_wheel(str(tmp_path), config_settings={"staging": str(stage)})


def test_companion_packaging_refuses_changed_staged_indexing(
    staged_wheel: tuple[Any, Path], tmp_path: Path
) -> None:
    backend, stage = staged_wheel
    source = stage / "phydrax_metal/_jax_indexing.py"
    source.write_bytes(source.read_bytes() + b"\n# staged input changed\n")
    with pytest.raises(backend.StagingError, match="does not match build.json"):
        backend.build_wheel(str(tmp_path), config_settings={"staging": str(stage)})
