#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import os
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
        env=dict(os.environ),
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


@pytest.mark.parametrize("status", ["failed", "timeout", "skipped", "unavailable", None])
def test_installed_qualification_refuses_nonexecuted_outcomes(
    status: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from importlib.machinery import ModuleSpec

    from tools import check_native_dependency_isolation as isolation

    scenarios = {"native-test": lambda options: {"status": status, "failure": "retained"}}

    def main() -> None:
        scenarios["native-test"](None)

    monkeypatch.setattr(isolation, "_block_qhull_operations", lambda: None)
    monkeypatch.setattr(
        isolation.importlib.util,
        "find_spec",
        lambda name: ModuleSpec(
            name, loader=None, origin=str(Path(sys.prefix) / name / "__init__.py")
        ),
    )
    monkeypatch.setattr(
        isolation.runpy,
        "run_module",
        lambda name: {"_SCENARIOS": scenarios, "main": main},
    )
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setitem(sys.modules, "tools", sys.modules["tools"])
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
        from tools.check_native_dependency_isolation import _refuse_external_process
        event, arguments = json.loads(sys.argv[1])
        sys.addaudithook(_refuse_external_process)
        try:
            if event == "subprocess.Popen":
                subprocess.run(arguments[1], executable=arguments[0], check=True)
            elif event == "os.posix_spawn":
                os.posix_spawn(arguments[0], arguments[1], dict(os.environ))
            else:
                os.system(arguments[0])
        except RuntimeError:
            raise SystemExit(0)
        raise SystemExit("External engine process was not refused.")
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
