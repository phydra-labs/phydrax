#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Install independent wheels and exercise native CAD and meshing without engines.

This is an execution/dependency gate, not evidence of unexercised native
numerical implementations. The consumer runs outside the checkout with Python
isolated mode, requires the installed in-house meshcore, and admits no optional
external geometry, meshing, partitioning or mesh-file codec package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_MODULES = (
    "OCP",
    "build123d",
    "meshio",
    "gmsh",
    "pygalmesh",
    "CGAL",
    "tetgen",
    "wildmeshing",
    "pymmg",
    "mmg",
    "parmmg",
    "omega_h",
    "manifold3d",
    "openvdb",
    "pyopenvdb",
    "open3d",
    "pyvista",
    "vtk",
    "vtkmodules",
    "pymetis",
    "metis",
    "meshpy",
    "igl",
    "libigl",
    "tioga",
    "pytioga",
)
EXTERNAL_EXECUTABLES = frozenset(
    (
        "gmsh",
        "mmg",
        "mmg2d",
        "mmg3d",
        "mmgs",
        "parmmg",
        "tetgen",
        "cgal",
        "tioga",
        "ftetwild",
        "tetwild",
        "vc_mesh",
    )
)


def _external(name: str, /) -> bool:
    return any(name == root or name.startswith(root + ".") for root in EXTERNAL_MODULES)


def _installed_external_modules() -> list[str]:
    """Report installed engines from ``sys.path`` without importing them.

    The isolated environment installs ``_RefuseExternal`` at startup, so the
    probe must bypass ``sys.meta_path``: an engine guard refuses every lookup,
    which says nothing about whether the package is installed.
    """
    return [
        name
        for name in EXTERNAL_MODULES
        if importlib.machinery.PathFinder.find_spec(name) is not None
    ]


class _RefuseExternal(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> None:
        if _external(fullname):
            raise ImportError(
                f"Native workflow attempted external engine import: {fullname}"
            )
        return None


def _refuse_external_process(event: str, arguments: tuple[object, ...], /) -> None:
    """Audit actual native-worker launches, including explicit shell commands."""
    if event not in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"):
        return
    executable = arguments[0]
    if isinstance(executable, bytes):
        executable = os.fsdecode(executable)
    if not isinstance(executable, str):
        raise RuntimeError(f"Uninspectable native process launch: {event}")
    tokens = shlex.split(executable) if event == "os.system" else [executable]
    if (
        Path(executable).name in ("sh", "bash", "zsh")
        and len(arguments) > 1
        and isinstance(arguments[1], (tuple, list))
    ):
        argv = arguments[1]
        if "-c" in argv:
            index = argv.index("-c") + 1
            if index < len(argv) and isinstance(argv[index], str):
                tokens.extend(shlex.split(argv[index]))
    forbidden = [
        token
        for token in tokens
        if Path(token).name.lower().removesuffix(".exe") in EXTERNAL_EXECUTABLES
    ]
    if forbidden:
        raise RuntimeError(
            f"Native workflow attempted external engine process: {forbidden}"
        )
    print(
        json.dumps({"native_process_launch": {"event": event, "executable": executable}}),
        file=sys.stderr,
        flush=True,
    )


def _refuse_qhull(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError("A native workflow attempted a SciPy/Qhull geometry operation.")


def _block_qhull_operations() -> None:
    # SciPy statistics legitimately imports scipy.spatial for distance metrics.
    # Block geometry execution, not this permitted numerical import dependency.
    import scipy.spatial as spatial

    qhull = sys.modules[spatial.Delaunay.__module__]
    for name in ("Delaunay", "ConvexHull", "Voronoi", "HalfspaceIntersection"):
        setattr(spatial, name, _refuse_qhull)
        setattr(qhull, name, _refuse_qhull)
    setattr(qhull, "_Qhull", _refuse_qhull)


def smoke_native_workflows(workspace: Path, /) -> dict[str, Any]:
    """Exercise format admission, exact CAD queries and CAD-derived native cells."""
    sys.meta_path.insert(0, _RefuseExternal())
    _block_qhull_operations()
    import numpy as np

    import phydrax as phx
    from phydrax._meshcore import (
        load_meshcore,
        meshcore_identity,
        meshcore_runtime_identity,
    )

    # This route requires the own kernel; retain the loader's exact ABI evidence.
    library = load_meshcore()
    if not library.path.resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(
            "The native binary does not resolve to the independent environment."
        )
    abi_contract = library["phx_mc_abi_contract"]()
    if not isinstance(abi_contract, bytes):
        raise RuntimeError(
            "The admitted native ABI getter did not return its contract bytes."
        )
    contract = phx.SpatialCoordinateContract.si()
    tessellation = phx.geometry.BRepTessellationPolicy(
        linear_deflection=0.1, angular_deflection=0.5
    )
    model = phx.geometry.brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        coordinate_contract=contract,
        tessellation=tessellation,
    )
    limits = phx.interchange.ResourceLimits(
        16 * 1024 * 1024, 64, 100_000, 1_000_000, 1024
    )
    policy = phx.interchange.CadImportPolicy(contract, limits, tessellation=tessellation)
    exports = (
        ("step", phx.interchange.write_step),
        ("iges", phx.interchange.write_iges),
        ("brep", phx.interchange.write_brep_text),
    )
    evidence: dict[str, Any] = {}
    decoded = model
    for format_name, writer in exports:
        path = workspace / f"box.{format_name}"
        writer(model, path)
        result = phx.interchange.read_cad(
            path.name,
            policy,
            trusted_root=workspace,
            source_length_unit=phx.units.METER if format_name == "brep" else None,
        )
        decoded = result.model
        if (decoded.topology.num_solids, decoded.topology.num_faces) != (1, 6):
            raise RuntimeError(f"{format_name}: box solid/face incidence changed.")
        containment = phx.geometry.prepare_brep_query(decoded).contains(
            np.asarray([[0.5, 1.0, 1.5], [2.0, 1.0, 1.5]], dtype=np.float64)
        )
        if np.asarray(containment.inside).tolist() != [True, False]:
            raise RuntimeError(
                f"{format_name}: native exact box containment is incorrect."
            )
        if np.any(
            np.asarray(containment.status) != phx.geometry.BRepProjectionStatus.UNIQUE
        ):
            raise RuntimeError(f"{format_name}: native box containment is unresolved.")
        evidence[format_name] = {
            "source_digest": decoded.source_digest,
            "model_id": decoded.model_id,
            "solid_count": decoded.topology.num_solids,
            "face_count": decoded.topology.num_faces,
            "containment_status": np.asarray(containment.status).tolist(),
        }
    archive = phx.interchange.save_brep_archive(decoded, workspace / "box.phxbrep")
    restored = phx.interchange.load_brep_archive(archive.path)
    if restored.model_id != decoded.model_id:
        raise RuntimeError(
            "Native lifecycle CAD persistence changed exact model identity."
        )
    domain = phx.geometry.MeshingDomain.from_brep(restored)
    meshing = phx.meshing
    scope = meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.source_indices[2], dtype=np.int64),
    )
    specification = meshing.SurfaceMeshingSpec(
        meshing.CellMeshingTarget(2, 3, meshing.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            meshing.UniformSizeControl(
                scope, 0.5, strength=meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    generated = (
        meshing.NativeMeshingProvider(meshing.NativeMeshingOptions("parametric_surface"))
        .plan(
            meshing.NativeSurfaceSource(domain),
            specification,
            coordinate_contract=contract,
        )
        .execute()
    )
    if not isinstance(generated, meshing.CellMeshingResult):
        raise RuntimeError("Native generation did not publish the canonical cell result.")
    points = np.asarray(generated.mesh.coordinates, dtype=np.float64)
    faces = np.asarray(generated.mesh.blocks[0].vertices, dtype=np.int64)
    corners = points[faces]
    area = float(
        np.sum(
            np.linalg.norm(
                np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                axis=1,
            )
        )
        / 2.0
    )
    volume = float(np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2])) / 6.0)
    np.testing.assert_allclose((area, volume), (22.0, 6.0), rtol=0.0, atol=1e-10)
    if generated.certification is None:
        raise RuntimeError(
            "Native CAD-derived generation omitted its certification evidence."
        )
    if not (
        generated.audit.passed
        and generated.compliance.passed
        and generated.certification.passed
    ):
        raise RuntimeError(
            "Native CAD-derived generation did not pass its publication evidence."
        )
    loaded = sorted(name for name in sys.modules if _external(name))
    if loaded:
        raise RuntimeError(f"External engines loaded: {loaded}")
    return {
        "phydrax_path": str(Path(phx.__file__).resolve()),
        "meshcore": meshcore_identity(),
        "meshcore_runtime": meshcore_runtime_identity(),
        "meshcore_binary_path": str(library.path.resolve()),
        "meshcore_abi_contract": abi_contract.decode("ascii"),
        "cad_formats": evidence,
        "surface_area": area,
        "enclosed_volume": volume,
        "external_modules_loaded": loaded,
        "scope": "native CAD codecs, exact box queries, persistence and CAD surface generation",
    }


def _run(command: Sequence[str], /, *, cwd: Path, env: dict[str, str]) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"{' '.join(command)} failed with exit code {result.returncode}."
        )


def _build_wheel(source: Path, destination: Path, /, *, env: dict[str, str]) -> Path:
    _run(
        ("uv", "build", "--wheel", "--out-dir", str(destination), str(source)),
        cwd=source,
        env=env,
    )
    (wheel,) = destination.glob("*.whl")
    return wheel


def _authenticate_wheels(wheels: Sequence[Path], /) -> None:
    """Match installed package payloads to the exact resolver input wheels."""
    evidence = {}
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.startswith(("phydrax/", "phydrax_meshcore/"))
                and not name.endswith("/")
            ]
            if not members:
                raise SystemExit(f"{wheel.name} contains no native package payload.")
            for name in members:
                package, relative = name.split("/", 1)
                spec = importlib.util.find_spec(package)
                if spec is None or spec.origin is None:
                    raise SystemExit(f"The installed wheel is missing {package}.")
                root = Path(spec.origin).resolve().parent
                if not root.is_relative_to(Path(sys.prefix).resolve()):
                    raise SystemExit(
                        f"{package} resolves outside the independent environment."
                    )
                installed = root / relative
                expected = hashlib.sha256(archive.read(name)).hexdigest()
                with installed.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != expected:
                    raise SystemExit(
                        f"Installed payload differs from {wheel.name}: {name}"
                    )
            evidence[wheel.name] = {"authenticated_payload_members": len(members)}
    print(
        json.dumps({"installed_wheel_authentication": evidence}, sort_keys=True),
        flush=True,
    )


def _qualification(
    workspace: Path, arguments: Sequence[str], /, *, wheels: Sequence[Path] = ()
) -> None:
    """Execute the existing driver against installed packages, never checkout imports."""
    for name in ("phydrax", "phydrax_meshcore"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise SystemExit(f"The independent environment is missing {name}.")
        if not Path(spec.origin).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise SystemExit(f"{name} does not resolve to the independent environment.")
    sys.meta_path.insert(0, _RefuseExternal())
    _block_qhull_operations()
    # Drivers are staged as an audit harness in this environment. No checkout
    # path or source package is inserted into the installed consumer's imports.
    sys.argv = ["meshing_qualification", *arguments]
    os.chdir(workspace)
    namespace = runpy.run_module("tools.meshing_qualification")
    scenarios = namespace["_SCENARIOS"]
    if list(arguments) == ["--all-native"]:
        commands = [
            ["--scenario", name]
            for name in scenarios
            if name.startswith("native-")
            and name
            not in (
                "native-corpus",
                "native-family-lifecycle",
                "native-lifecycle",
                "native-polyhedral-lifecycle",
                "native-cad-lifecycle",
                "native-periodic-lifecycle",
            )
        ]
        profiles = namespace["_family_qualification_requirements"]()["parser_choices"]
        commands.extend(
            ["--scenario", "native-family-lifecycle", "--family-case", name]
            for name in profiles
        )
        commands.extend(
            ["--scenario", "native-lifecycle", "--corpus-case", case.name]
            for case in namespace["CORPUS"]
            if case.expected_admission == "positive"
        )
        commands.extend(
            ["--scenario", "native-polyhedral-lifecycle", "--polyhedral-source", name]
            for name in namespace["POLYHEDRAL_CASE_NAMES"]
        )
        commands.extend(
            ["--scenario", "native-cad-lifecycle", "--cad-format", format_name]
            for format_name in ("step", "iges", "brep")
        )
        commands.extend(
            ["--scenario", "native-periodic-lifecycle", "--periodic-case", case]
            for case in ("skew-lattice", "skew-material-feature")
        )
        failures = []
        for command in commands:
            try:
                _run(
                    (
                        sys.executable,
                        "-I",
                        str(Path(__file__).resolve()),
                        "--consumer",
                        str(workspace),
                        "--qualification-consumer",
                        "--qualification",
                        json.dumps(command),
                        *(
                            (
                                "--wheel",
                                str(wheels[0]),
                                "--meshcore-wheel",
                                str(wheels[1]),
                            )
                            if wheels
                            else ()
                        ),
                    ),
                    cwd=workspace,
                    env=dict(os.environ),
                )
            except SystemExit as failure:
                failures.append(str(failure))
        if failures:
            raise SystemExit("\n\n".join(failures))
        return
    scenario = arguments[arguments.index("--scenario") + 1]
    original = scenarios[scenario]

    def require_execution(options: argparse.Namespace) -> dict[str, Any]:
        record = original(options)
        if record.get("status") != "passed":
            print(json.dumps(record, indent=2, sort_keys=True))
            raise SystemExit(
                f"{scenario}: qualification did not report passed execution."
            )
        return record

    scenarios[scenario] = require_execution
    namespace["main"]()
    loaded = sorted(name for name in sys.modules if _external(name))
    if loaded:
        raise RuntimeError(f"External engines loaded during qualification: {loaded}")


def _consumer(workspace: Path, /) -> None:
    installed = _installed_external_modules()
    if installed:
        raise SystemExit(f"External engine/codec packages installed: {installed}")
    for name in ("phydrax", "phydrax_meshcore"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise SystemExit(f"The independent environment is missing {name}.")
        if not Path(spec.origin).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise SystemExit(f"{name} does not resolve to the independent environment.")
    print(json.dumps(smoke_native_workflows(workspace), indent=2, sort_keys=True))


def _missing_meshcore_consumer() -> None:
    """Prove neutral import and truthful refusal, not native construction."""
    sys.addaudithook(_refuse_external_process)
    if importlib.util.find_spec("phydrax_meshcore") is not None:
        raise SystemExit(
            "The missing-meshcore environment unexpectedly contains meshcore."
        )
    installed = _installed_external_modules()
    if installed:
        raise SystemExit(f"External engine/codec packages installed: {installed}")
    sys.meta_path.insert(0, _RefuseExternal())
    import phydrax as phx
    from phydrax._meshcore import (
        load_meshcore,
        meshcore_available,
        MeshcoreUnavailableError,
    )

    if not Path(phx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise SystemExit("Base import does not resolve to the independent environment.")
    if meshcore_available():
        raise SystemExit("Missing meshcore was incorrectly reported available.")
    try:
        load_meshcore()
    except MeshcoreUnavailableError as refusal:
        print(
            json.dumps(
                {
                    "scope": "neutral base import and missing-kernel refusal only",
                    "phydrax_path": str(Path(phx.__file__).resolve()),
                    "meshcore_available": False,
                    "refusal": str(refusal),
                    "native_construction_exercised": False,
                },
                sort_keys=True,
            )
        )
    else:
        raise SystemExit(
            "Missing meshcore did not produce its canonical availability refusal."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--meshcore-wheel", type=Path)
    parser.add_argument("--python", default="3.12")
    parser.add_argument("--consumer", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--missing-meshcore-consumer", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--qualification",
        action="append",
        default=[],
        metavar="JSON_ARGUMENT_ARRAY",
        help="Run an existing native qualification command in the installed environment; "
        "repeat for each mandatory workflow. Example: "
        """'["--scenario","native-surface-pde"]'""",
    )
    parser.add_argument(
        "--qualification-consumer", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--all-native",
        action="store_true",
        help="Execute every native driver scenario and every declared family profile; "
        "fail rather than skip any unavailable or incomplete workflow.",
    )
    arguments = parser.parse_args()
    if arguments.consumer is not None:
        wheels = tuple(
            path.resolve()
            for path in (arguments.wheel, arguments.meshcore_wheel)
            if path is not None
        )
        if wheels:
            if len(wheels) != (1 if arguments.missing_meshcore_consumer else 2):
                raise SystemExit(
                    "Installed authentication requires the selected environment's wheels."
                )
            _authenticate_wheels(wheels)
        if arguments.missing_meshcore_consumer:
            _missing_meshcore_consumer()
        elif arguments.qualification_consumer:
            command = json.loads(arguments.qualification[0])
            _qualification(arguments.consumer, command, wheels=wheels)
        else:
            _consumer(arguments.consumer)
        return
    environment = {
        name: value
        for name, value in os.environ.items()
        if name
        not in ("PYTHONPATH", "PYTHONHOME", "PHYDRAX_MESHCORE_LIBRARY", "VIRTUAL_ENV")
    }
    with tempfile.TemporaryDirectory(prefix="phydrax-native-isolation-") as temporary:
        workspace = Path(temporary)
        wheel = (
            arguments.wheel.resolve()
            if arguments.wheel is not None
            else _build_wheel(ROOT, workspace / "dist", env=environment)
        )
        meshcore = (
            arguments.meshcore_wheel.resolve()
            if arguments.meshcore_wheel is not None
            else _build_wheel(
                ROOT / "native" / "meshcore", workspace / "native-dist", env=environment
            )
        )
        wheel_evidence = {}
        for name, path in (("phydrax", wheel), ("phydrax_meshcore", meshcore)):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            wheel_evidence[name] = {"path": str(path), "sha256": digest}
        print(
            json.dumps({"installation_inputs": wheel_evidence}, sort_keys=True),
            flush=True,
        )
        venv = workspace / "venv"
        _run(
            ("uv", "venv", "--python", arguments.python, str(venv)),
            cwd=workspace,
            env=environment,
        )
        python = venv / "bin" / "python"
        _run(
            ("uv", "pip", "install", "--python", str(python), str(wheel), str(meshcore)),
            cwd=workspace,
            env=environment,
        )
        (site_packages,) = (venv / "lib").glob("python*/site-packages")
        for namespace in ("tools", "examples"):
            shutil.copytree(
                ROOT / namespace,
                site_packages / namespace,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        # The existing drivers spawn genuine cold subprocesses. Enforce the same
        # denial boundary there rather than trusting only the outer process.
        (site_packages / "sitecustomize.py").write_text(
            "import contextlib, os, sys\n"
            "from pathlib import Path\n"
            "try:\n"
            "    from tools.check_native_dependency_isolation import "
            "_RefuseExternal, _block_qhull_operations, _authenticate_wheels, _refuse_external_process\n"
            "    sys.meta_path.insert(0, _RefuseExternal())\n"
            "    sys.addaudithook(_refuse_external_process)\n"
            "    with contextlib.redirect_stdout(sys.stderr):\n"
            f"        _authenticate_wheels((Path({str(wheel)!r}), Path({str(meshcore)!r})))\n"
            "    _block_qhull_operations()\n"
            "except BaseException as error:\n"
            "    print(f'Isolation startup refusal: {error}', file=sys.stderr, flush=True)\n"
            "    os._exit(1)\n",
            encoding="utf-8",
        )
        runner = site_packages / "tools" / Path(__file__).name
        consumer = workspace / "consumer"
        consumer.mkdir()
        _run(
            (
                str(python),
                "-I",
                str(runner),
                "--consumer",
                str(consumer),
                "--wheel",
                str(wheel),
                "--meshcore-wheel",
                str(meshcore),
            ),
            cwd=consumer,
            env=environment,
        )
        base = workspace / "base-venv"
        _run(
            ("uv", "venv", "--python", arguments.python, str(base)),
            cwd=workspace,
            env=environment,
        )
        base_python = base / "bin" / "python"
        _run(
            ("uv", "pip", "install", "--python", str(base_python), str(wheel)),
            cwd=workspace,
            env=environment,
        )
        _run(
            (
                str(base_python),
                "-I",
                str(runner),
                "--consumer",
                str(consumer),
                "--missing-meshcore-consumer",
                "--wheel",
                str(wheel),
            ),
            cwd=consumer,
            env=environment,
        )
        if arguments.all_native:
            arguments.qualification.append(json.dumps(["--all-native"]))
        for encoded in arguments.qualification:
            command = json.loads(encoded)
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(value, str) for value in command)
            ):
                raise ValueError("--qualification requires a nonempty JSON string array.")
            if command != ["--all-native"]:
                if "--scenario" not in command or command.index("--scenario") + 1 >= len(
                    command
                ):
                    raise ValueError(
                        "Qualification commands must explicitly select a native scenario."
                    )
                scenario = command[command.index("--scenario") + 1]
                if not scenario.startswith("native-"):
                    raise ValueError(
                        "Dependency isolation accepts only native qualification scenarios."
                    )
            _run(
                (
                    str(python),
                    "-I",
                    str(runner),
                    "--consumer",
                    str(consumer),
                    "--qualification-consumer",
                    "--qualification",
                    encoded,
                    "--wheel",
                    str(wheel),
                    "--meshcore-wheel",
                    str(meshcore),
                ),
                cwd=consumer,
                env=environment,
            )


if __name__ == "__main__":
    main()
