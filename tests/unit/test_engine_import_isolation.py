#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_base_imports_and_neutral_contracts_load_no_cad_engine_or_mesh_codec() -> None:
    # Any attempted import of a blocked engine raises, so a hidden eager edge fails
    # here even if the engine is installed.
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        BLOCKED = (
            "OCP", "build123d", "meshio", "gmsh", "pygalmesh", "CGAL",
            "tetgen", "wildmeshing", "pymmg", "omega_h", "manifold3d",
            "openvdb", "pyopenvdb", "open3d", "pyvista", "vtk", "vtkmodules",
            "pymetis", "metis", "meshpy", "igl", "libigl", "tioga", "pytioga",
        )

        def blocked(name):
            return any(name == root or name.startswith(root + ".") for root in BLOCKED)

        class Refuse(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if blocked(name):
                    raise ImportError(f"base import loaded {name}")
                return None

        sys.meta_path.insert(0, Refuse())

        import numpy as np

        import phydrax as phx
        import phydrax.discretization
        import phydrax.geometry
        import phydrax.meshing
        import phydrax.interchange
        from phydrax.interchange import CadImportPolicy, read_cad
        from phydrax.geometry.simplicial._io import triangle_arrays

        embedding = phx.geometry.PlanarEmbedding(
            (1.0, 2.0, 3.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
        )
        world = embedding.to_world(np.asarray([[0.5, -2.0]]))
        assert np.allclose(world, [[1.0, 2.5, 1.0]])
        assert np.allclose(embedding.to_planar(world), [[0.5, -2.0]])
        policy = phx.geometry.BRepProjectionPolicy()
        status = phx.geometry.BRepProjectionStatus.SEAM
        assert policy.ambiguity_tolerance > 0.0 and int(status) == 1
        vertices, faces = triangle_arrays(
            (np.eye(3, dtype=np.float64), np.asarray([[0, 1, 2]], dtype=np.int32))
        )
        assert vertices.shape == (3, 3) and faces.shape == (1, 3)
        assert not [name for name in sys.modules if blocked(name)]
        print("isolated")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("isolated")


@pytest.mark.parametrize("suffix", [".step", ".iges", ".brep"])
def test_native_cad_dispatch_retains_codec_byte_budget(
    suffix: str, tmp_path: Path
) -> None:
    import phydrax as phx

    source = tmp_path / f"oversized{suffix}"
    source.write_bytes(b"not a CAD file")
    policy = phx.interchange.CadImportPolicy(
        phx.SpatialCoordinateContract.si(),
        phx.interchange.ResourceLimits(1, 64, 100, 100, 10),
    )
    with pytest.raises(phx.interchange.CadInterchangeError) as refusal:
        phx.interchange.read_cad(
            source.name,
            policy,
            trusted_root=tmp_path,
            source_length_unit=phx.units.METER if suffix == ".brep" else None,
        )
    assert refusal.value.refusal.reason == "limit"


@pytest.mark.parametrize("suffix", [".step", ".iges", ".brep"])
def test_native_cad_dispatch_retains_malformed_format_refusal(
    suffix: str, tmp_path: Path
) -> None:
    import phydrax as phx

    source = tmp_path / f"malformed{suffix}"
    source.write_bytes(b"not a CAD file")
    policy = phx.interchange.CadImportPolicy(
        phx.SpatialCoordinateContract.si(),
        phx.interchange.ResourceLimits(1024, 64, 100, 100, 10),
    )
    with pytest.raises(phx.interchange.CadInterchangeError) as refusal:
        phx.interchange.read_cad(
            source.name,
            policy,
            trusted_root=tmp_path,
            source_length_unit=phx.units.METER if suffix == ".brep" else None,
        )
    assert refusal.value.refusal.reason == "malformed"


@pytest.mark.parametrize("suffix", [".step", ".iges"])
def test_native_cad_embedded_units_cannot_be_overridden(
    suffix: str, tmp_path: Path
) -> None:
    import phydrax as phx

    policy = phx.interchange.CadImportPolicy(
        phx.SpatialCoordinateContract.si(),
        phx.interchange.ResourceLimits(1024, 64, 100, 100, 10),
    )
    with pytest.raises(phx.interchange.CadInterchangeError) as refusal:
        phx.interchange.read_cad(
            f"box{suffix}",
            policy,
            trusted_root=tmp_path,
            source_length_unit=phx.units.METER,
        )
    assert refusal.value.refusal.reason == "units"


def test_native_brep_text_requires_declared_source_units(tmp_path: Path) -> None:
    import phydrax as phx

    policy = phx.interchange.CadImportPolicy(
        phx.SpatialCoordinateContract.si(),
        phx.interchange.ResourceLimits(1024, 64, 100, 100, 10),
    )
    with pytest.raises(phx.interchange.CadInterchangeError) as refusal:
        phx.interchange.read_cad("box.brep", policy, trusted_root=tmp_path)
    assert refusal.value.refusal.reason == "units"


@pytest.mark.parametrize(
    "status", ["failed", "timeout", "skipped", "unavailable", "blocked", "refused", None]
)
def test_installed_qualification_refuses_nonexecuted_outcomes(
    status: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from importlib.machinery import ModuleSpec
    from types import ModuleType

    from tools import check_native_dependency_isolation as isolation
    from tools.meshing_qualification._controls import (
        QualificationControls,
        SCENARIO_MODULES,
    )

    class RefusedScenario(ModuleType):
        def run(self, options: QualificationControls) -> dict[str, object]:
            del options
            return {"status": status, "failure": "retained"}

    scenario = RefusedScenario("tools.meshing_qualification.refused_test_scenario")
    monkeypatch.setitem(sys.modules, scenario.__name__, scenario)
    monkeypatch.setitem(SCENARIO_MODULES, "native-test", "refused_test_scenario")
    find_spec = isolation.importlib.util.find_spec

    def installed_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name in ("phydrax", "phydrax_meshcore"):
            return ModuleSpec(
                name, loader=None, origin=str(Path(sys.prefix) / name / "__init__.py")
            )
        return find_spec(name, package)

    monkeypatch.setattr(isolation, "_block_qhull_operations", lambda: None)
    monkeypatch.setattr(isolation.importlib.util, "find_spec", installed_spec)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="did not report passed execution"):
        isolation._qualification(tmp_path, ["--scenario", "native-test"])


@pytest.mark.parametrize(
    "member", ["phydrax/__init__.py", "phydrax_meshcore/libmeshcore.dylib"]
)
def test_installed_wheel_authentication_detects_payload_replacement(
    member: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zipfile
    from importlib.machinery import ModuleSpec

    from tools import check_native_dependency_isolation as isolation

    wheel = tmp_path / "payload.whl"
    payload = b"original exact wheel bytes"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(member, payload)
    installed = tmp_path / member
    installed.parent.mkdir()
    installed.write_bytes(payload)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(
        isolation.importlib.util,
        "find_spec",
        lambda name: ModuleSpec(
            name, loader=None, origin=str(tmp_path / name / "__init__.py")
        ),
    )
    isolation._authenticate_wheels((wheel,))
    installed.write_bytes(b"replacement")
    with pytest.raises(SystemExit, match="Installed payload differs"):
        isolation._authenticate_wheels((wheel,))


def test_installed_wheel_authentication_refuses_checkout_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zipfile
    from importlib.machinery import ModuleSpec

    from tools import check_native_dependency_isolation as isolation

    wheel = tmp_path / "payload.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("phydrax/__init__.py", b"payload")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(
        isolation.importlib.util,
        "find_spec",
        lambda name: ModuleSpec(
            name, loader=None, origin=str(tmp_path / name / "__init__.py")
        ),
    )
    with pytest.raises(SystemExit, match="outside the independent environment"):
        isolation._authenticate_wheels((wheel,))


@pytest.mark.parametrize(
    ("event", "arguments"),
    [
        ("subprocess.Popen", ("/usr/local/bin/gmsh", ["gmsh", "input.geo"])),
        ("os.posix_spawn", ("/usr/local/bin/mmg3d", ["mmg3d", "input.mesh"])),
        ("os.system", ("tetgen input.poly",)),
        ("subprocess.Popen", ("/bin/sh", ["/bin/sh", "-c", "gmsh input.geo"])),
    ],
)
def test_native_execution_audit_refuses_external_engine_process(
    event: str, arguments: tuple[object, ...]
) -> None:
    code = textwrap.dedent(
        """
        import json, os, subprocess, sys
        import pytest
        from tools.check_native_dependency_isolation import _refuse_external_process
        event, arguments = json.loads(sys.argv[1])
        sys.addaudithook(_refuse_external_process)
        with pytest.raises(RuntimeError):
            if event == "subprocess.Popen":
                subprocess.run(arguments[1], executable=arguments[0], check=True)
            elif event == "os.posix_spawn":
                os.posix_spawn(arguments[0], arguments[1], dict(os.environ))
            else:
                os.system(arguments[0])
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps((event, arguments))],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_native_execution_audit_allows_real_python_worker_arguments() -> None:
    code = textwrap.dedent(
        """
        import subprocess, sys
        from tools.check_native_dependency_isolation import _refuse_external_process
        sys.addaudithook(_refuse_external_process)
        result = subprocess.run(
            [sys.executable, "-c", "print('native')", "gmsh"],
            capture_output=True, text=True, check=True,
        )
        if result.stdout.strip() != "native":
            raise SystemExit("Native Python worker did not execute.")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_staged_harness_loads_reviewed_inputs_and_runtime_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import importlib

    from tools import check_native_dependency_isolation as isolation
    from tools._meshing_cases import (
        load_cad_fixture,
        load_corpus_registration,
        load_curved_periodic_source_registration,
    )

    site_packages = tmp_path / "site-packages"
    evidence = isolation._stage_harness(site_packages)
    for relative, digest in evidence.items():
        staged = site_packages / relative
        assert staged.read_bytes() == (isolation.ROOT / relative).read_bytes()
        assert hashlib.sha256(staged.read_bytes()).hexdigest() == digest
        assert staged.stat().st_mode & 0o222 == 0
    cad = load_cad_fixture(site_packages / "tests/data/cad/native_authored_boxes.json")
    assert cad["file_sha256"] == evidence["tests/data/cad/native_authored_boxes.json"]
    curved = load_curved_periodic_source_registration(
        site_packages / "tests/data/meshing/curved_periodic_sources.json"
    )
    assert (
        curved["file_sha256"]
        == evidence["tests/data/meshing/curved_periodic_sources.json"]
    )
    assert (
        load_corpus_registration(site_packages / "tests/data/meshing/corpus.json")
        == load_corpus_registration()
    )
    monkeypatch.syspath_prepend(str(site_packages))
    for name in ("benchmarks", "benchmarks._runtime"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    runtime = importlib.import_module("benchmarks._runtime")
    monkeypatch.delitem(sys.modules, "benchmarks._runtime")
    monkeypatch.delitem(sys.modules, "benchmarks")
    assert Path(runtime.__file__).is_relative_to(site_packages)
    assert runtime.logical_array_bytes({"field": __import__("numpy").zeros(3)}) == 24
    assert not (site_packages / "phydrax").exists()
    assert not (site_packages / "tests/__init__.py").exists()


def _installed_source_build_tree(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "source"
    site_packages = tmp_path / "site-packages"
    inputs = {
        "phydrax/__init__.py": b"VALUE = 1\n",
        "phydrax/model.py": b"MODEL = 2\n",
        "pyproject.toml": b"[project]\nname = 'authored-source'\n",
        "uv.lock": b"version = 1\n",
        "native/meshcore/src/kernel.cpp": b"int kernel() { return 1; }\n",
        "native/meshcore/include/kernel.h": b"int kernel();\n",
        "native/meshcore/tests/test_kernel.cpp": b"int main() { return 0; }\n",
        "native/meshcore/python/phydrax_meshcore/__init__.py": b"ABI = 1\n",
        "native/meshcore/NOTICE": b"authored native source notice\n",
        "native/providers/example/worker.cpp": b"int worker() { return 2; }\n",
    }
    for relative, payload in inputs.items():
        source = source_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(payload)
        if relative.startswith("phydrax/"):
            installed = site_packages / relative
            installed.parent.mkdir(parents=True, exist_ok=True)
            installed.write_bytes(payload)
    return source_root, site_packages


def test_installed_harness_authenticates_complete_source_build_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import hashlib

    from benchmarks._runtime import source_build_fingerprint
    from tools import check_native_dependency_isolation as isolation

    source_root, site_packages = _installed_source_build_tree(tmp_path)
    ignored = source_root / "native/meshcore/build/generated.cpp"
    ignored.parent.mkdir()
    ignored.write_bytes(b"not an authored build input\n")
    expected = source_build_fingerprint(source_root)
    evidence = isolation._stage_source_build_inputs(source_root, site_packages)

    assert set(evidence) == {
        "pyproject.toml",
        "uv.lock",
        "native/meshcore/src/kernel.cpp",
        "native/meshcore/include/kernel.h",
        "native/meshcore/tests/test_kernel.cpp",
        "native/meshcore/python/phydrax_meshcore/__init__.py",
        "native/meshcore/NOTICE",
        "native/providers/example/worker.cpp",
    }
    for relative, digest in evidence.items():
        staged = site_packages / relative
        assert staged.read_bytes() == (source_root / relative).read_bytes()
        assert hashlib.sha256(staged.read_bytes()).hexdigest() == digest
        assert staged.stat().st_mode & 0o222 == 0
    assert not (site_packages / "native/meshcore/build").exists()
    assert (site_packages / "phydrax/model.py").read_bytes() == b"MODEL = 2\n"
    assert source_build_fingerprint(site_packages) == expected
    assert json.loads(capsys.readouterr().out) == {
        "authenticated_harness_source_build_fingerprint": expected,
        "staged_harness_source_build_inputs_sha256": evidence,
    }


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra"])
def test_installed_harness_refuses_mismatched_package_source_inventory(
    mutation: str, tmp_path: Path
) -> None:
    from tools import check_native_dependency_isolation as isolation

    source_root, site_packages = _installed_source_build_tree(tmp_path)
    installed = site_packages / "phydrax/model.py"
    if mutation == "changed":
        installed.write_bytes(b"MODEL = 3\n")
    elif mutation == "missing":
        installed.unlink()
    else:
        (site_packages / "phydrax/extra.py").write_bytes(b"EXTRA = 1\n")
    before = {
        path.relative_to(site_packages).as_posix(): path.read_bytes()
        for path in (site_packages / "phydrax").rglob("*.py")
    }
    with pytest.raises(SystemExit):
        isolation._stage_source_build_inputs(source_root, site_packages)
    assert {
        path.relative_to(site_packages).as_posix(): path.read_bytes()
        for path in (site_packages / "phydrax").rglob("*.py")
    } == before


@pytest.mark.parametrize("missing", ["pyproject.toml", "uv.lock"])
def test_installed_harness_requires_authored_build_inputs(
    missing: str, tmp_path: Path
) -> None:
    from tools import check_native_dependency_isolation as isolation

    source_root, site_packages = _installed_source_build_tree(tmp_path)
    (source_root / missing).unlink()
    with pytest.raises(FileNotFoundError):
        isolation._stage_source_build_inputs(source_root, site_packages)
    assert not (site_packages / "pyproject.toml").exists()
    assert not (site_packages / "uv.lock").exists()


def test_installed_harness_refuses_corrupted_staged_build_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import check_native_dependency_isolation as isolation

    source_root, site_packages = _installed_source_build_tree(tmp_path)
    write_bytes = Path.write_bytes

    def corrupted_write(path: Path, payload: bytes) -> int:
        if path == site_packages / "uv.lock":
            payload = b"replaced authored lock\n"
        return write_bytes(path, payload)

    monkeypatch.setattr(Path, "write_bytes", corrupted_write)
    with pytest.raises(RuntimeError):
        isolation._stage_source_build_inputs(source_root, site_packages)


def test_installed_harness_rejects_unauthenticated_wheel_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zipfile

    from tools import check_native_dependency_isolation as isolation

    source_root, site_packages = _installed_source_build_tree(tmp_path)
    wheel = tmp_path / "phydrax.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("unrelated.txt", b"no authenticated package payload")
    monkeypatch.setattr(isolation, "ROOT", site_packages)
    with pytest.raises(SystemExit):
        isolation.main(
            [
                "--stage-source-build-inputs",
                str(source_root),
                "--wheel",
                str(wheel),
                "--meshcore-wheel",
                str(tmp_path / "phydrax_meshcore.whl"),
            ]
        )
    assert not (site_packages / "pyproject.toml").exists()
    assert not (site_packages / "uv.lock").exists()
    assert not (site_packages / "native").exists()


def test_distributed_consumer_checkpoint_is_fresh_and_persists_output(
    tmp_path: Path,
) -> None:
    from tools import check_native_dependency_isolation as isolation

    first = isolation._fresh_distributed_checkpoint(tmp_path)
    assert first.is_relative_to(tmp_path)
    assert first.parent.is_dir()
    assert not first.exists()
    first.mkdir()
    receipt = first / "producer-receipt"
    receipt.write_bytes(b"durable producer output")
    second = isolation._fresh_distributed_checkpoint(tmp_path)
    assert second != first
    assert not second.exists()
    assert receipt.read_bytes() == b"durable producer output"
