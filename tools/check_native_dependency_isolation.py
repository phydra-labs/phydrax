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
import contextlib
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple, NoReturn


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
# Launch forms the process guard follows to the programs they execute. Any other
# program is inspected by its invoked and resolved names only: its arguments are
# data, and what an admitted program launches after it starts is not observed.
# The guard keeps external engines out of this harness's evidence; it is not a
# sandbox.
_SHELLS = frozenset(("ash", "bash", "dash", "ksh", "mksh", "sh", "zsh"))
_SHELL_LONG_OPTIONS = frozenset(
    ("--login", "--noediting", "--noprofile", "--norc", "--posix", "--restricted")
)
# Launchers whose command grammars the guard does not parse: an engine named by
# any argument word refuses the launch, and a shell or env among their arguments
# is unsupported because its role cannot be read.
_OPAQUE_LAUNCHERS = frozenset(
    (
        "busybox",
        "chroot",
        "chrt",
        "doas",
        "flock",
        "ionice",
        "mpiexec",
        "mpirun",
        "nice",
        "nohup",
        "nsenter",
        "numactl",
        "prlimit",
        "runuser",
        "setpriv",
        "setsid",
        "srun",
        "stdbuf",
        "strace",
        "sudo",
        "systemd-run",
        "taskset",
        "time",
        "timeout",
        "unshare",
        "uv",
        "uvx",
        "valgrind",
        "xargs",
    )
)
_LAUNCH_FORMS = frozenset(("env", *_SHELLS, *_OPAQUE_LAUNCHERS))
# Shell text whose words cannot be read literally: parameter, command, tilde
# and brace expansion and globs can produce any program path at run time.
_NONLITERAL_SHELL_TEXT = frozenset("$`*?[{~")
_QUOTING = str.maketrans("", "", "'\"\\")
_WORD_SEPARATOR = re.compile(r"[\s;&|()<>=]+")


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
                f"native_isolation.import.module must exclude external engines; got {fullname!r}"
            )
        return None


class _LaunchContext(NamedTuple):
    """Directory and environment a launched program name resolves against."""

    cwd: str
    environment: dict[str, str]


def _launch_refusal(form: str, /) -> RuntimeError:
    return RuntimeError(
        f"native_isolation.process.launch must use a supported inspectable form; got {form}"
    )


def _launch_text(value: object, /) -> str:
    if isinstance(value, (str, bytes, os.PathLike)):
        return os.fsdecode(value)
    raise RuntimeError(
        f"native_isolation.process.argument must be inspectable text; got {value!r}"
    )


def _launch_environment(environment: object, /) -> dict[str, str]:
    source = os.environ if environment is None else environment
    if not isinstance(source, Mapping):
        raise RuntimeError(
            f"native_isolation.process.environment must be an inspectable mapping; got {type(source).__name__}"
        )
    return {_launch_text(name): _launch_text(value) for name, value in source.items()}


def _program_names(program: str, context: _LaunchContext, /) -> tuple[str, ...]:
    """Return a program's invoked name, then the name of the file it resolves to.

    Resolution follows the launch's search path and symbolic links, including
    descriptor paths such as ``/proc/self/fd/N``.
    """
    if os.sep in program or (os.altsep is not None and os.altsep in program):
        located = os.path.join(context.cwd, program)
    else:
        located = shutil.which(
            program, path=os.pathsep.join(os.get_exec_path(context.environment))
        )
    names = (program,) if located is None else (program, os.path.realpath(located))
    return tuple(Path(name).name.lower().removesuffix(".exe") for name in names)


def _word_engines(text: str, context: _LaunchContext, /) -> set[str]:
    """Return engines named by any word of text, each read as a possible program."""
    return {
        name
        for word in _WORD_SEPARATOR.split(text.translate(_QUOTING))
        if word
        for name in _program_names(word, context)
        if name in EXTERNAL_EXECUTABLES
    }


def _shell_text_engines(text: str, context: _LaunchContext, /) -> set[str]:
    """Read shell command text conservatively: every literal word may run."""
    nonliteral = sorted(_NONLITERAL_SHELL_TEXT.intersection(text))
    if nonliteral:
        raise _launch_refusal(
            f"shell text with nonliteral words {nonliteral!r}: {text!r}"
        )
    return _word_engines(text, context)


def _shell_engines(arguments: Sequence[str], context: _LaunchContext, /) -> set[str]:
    """Read a POSIX shell's ``-c`` text; refuse shells reading unseen commands."""
    command = False
    index = 0
    while index < len(arguments) and arguments[index][:1] in ("-", "+"):
        argument = arguments[index]
        index += 1
        if argument in ("-", "--"):
            break
        if argument.startswith("--"):
            if argument not in _SHELL_LONG_OPTIONS:
                raise _launch_refusal(f"shell option {argument!r}")
            continue
        for option in argument[1:]:
            if option in ("o", "O"):
                index += 1
            elif option == "c" and argument[0] == "-":
                command = True
            elif option == "s" or not option.isalpha():
                raise _launch_refusal(f"shell option {argument!r}")
    if index > len(arguments):
        raise _launch_refusal("a shell option without its operand")
    if index == len(arguments):
        raise _launch_refusal(
            "shell -c without command text"
            if command
            else "a shell reading commands from standard input"
        )
    if command:
        return _shell_text_engines(arguments[index], context)
    # A script operand is a program by name, like a directly executed script.
    return {
        name
        for name in _program_names(arguments[index], context)
        if name in EXTERNAL_EXECUTABLES
    }


def _env_engines(arguments: Sequence[str], context: _LaunchContext, /) -> set[str]:
    """Follow ``env`` options and assignments to the command it executes."""
    environment = dict(context.environment)
    cwd = context.cwd
    index = 0
    while index < len(arguments) and arguments[index].startswith("-"):
        argument = arguments[index]
        index += 1
        if argument == "--":
            break
        if argument == "-":
            environment.clear()
            continue
        # An option with an operand: attached text, or None for the next argument.
        operand: tuple[str, str | None] | None = None
        if argument.startswith("--"):
            name, separator, value = argument.partition("=")
            if name in ("--unset", "--chdir"):
                operand = (name, value if separator else None)
            elif name == "--ignore-environment" and not separator:
                environment.clear()
            elif not (
                (name in ("--null", "--debug") and not separator)
                or name in ("--block-signal", "--default-signal", "--ignore-signal")
            ):
                raise _launch_refusal(f"env option {argument!r}")
        else:
            for position, option in enumerate(argument[1:], start=2):
                if option == "i":
                    environment.clear()
                elif option in ("u", "C"):
                    operand = (option, argument[position:] or None)
                    break
                elif option not in ("0", "v"):
                    raise _launch_refusal(f"env option {argument!r}")
        if operand is not None:
            option, value = operand
            if value is None:
                if index == len(arguments):
                    raise _launch_refusal(f"env option {argument!r} without its operand")
                value = arguments[index]
                index += 1
            if option in ("u", "--unset"):
                environment.pop(value, None)
            else:
                cwd = os.path.join(cwd, value)
    while index < len(arguments) and "=" in arguments[index]:
        name, _, value = arguments[index].partition("=")
        environment[name] = value
        index += 1
    if index == len(arguments):
        # Without a command, env reports its environment and runs no program.
        return set()
    command = arguments[index:]
    return _launched_engines(command, command[0], _LaunchContext(cwd, environment))


def _opaque_engines(arguments: Sequence[str], context: _LaunchContext, /) -> set[str]:
    """Refuse engines named anywhere among an unparsed launcher's arguments."""
    engines: set[str] = set()
    for argument in arguments:
        if any(
            name == "env" or name in _SHELLS for name in _program_names(argument, context)
        ):
            raise _launch_refusal(
                f"a shell or env behind an unparsed launcher: {argument!r}"
            )
        engines |= _word_engines(argument, context)
    return engines


def _launched_engines(
    argv: Sequence[str], program: str, context: _LaunchContext, /
) -> set[str]:
    """Return the external engines one supported launch form executes."""
    names = _program_names(program, context)
    engines = {name for name in names if name in EXTERNAL_EXECUTABLES}
    # A multi-call binary dispatches on its invoked name; an alias runs its target.
    form = next((name for name in names if name in _LAUNCH_FORMS), None)
    if engines or form is None:
        return engines
    if form == "env":
        return _env_engines(argv[1:], context)
    if form in _SHELLS:
        return _shell_engines(argv[1:], context)
    return _opaque_engines(argv[1:], context)


def _refuse_external_process(event: str, arguments: tuple[object, ...], /) -> None:
    """Refuse external engines in the launch forms this harness declares.

    A direct program is checked by its invoked name and by the name of the file
    it resolves to through the search path, symbolic links and descriptor paths.
    ``env`` options and assignments are followed to their command; POSIX shell
    ``-c`` and ``os.system`` text is read as literal words, each a possible
    program; a declared opaque launcher refuses any engine word. Arguments of
    any other program are data, and forms the guard cannot read are refused.
    This guards the harness's evidence, not the host: an engine copied under
    another name, or launched later by an admitted program, is not observed.
    """
    if event not in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"):
        return
    executable = _launch_text(arguments[0])
    if event == "os.system":
        context = _LaunchContext(os.getcwd(), _launch_environment(None))
        engines = _shell_text_engines(executable, context)
    else:
        argv = arguments[1]
        if not isinstance(argv, (list, tuple)):
            raise RuntimeError(
                f"native_isolation.process.arguments must be an inspectable sequence; got event={event!r}, arguments={argv!r}"
            )
        popen = event == "subprocess.Popen"
        cwd = arguments[2] if popen else None
        context = _LaunchContext(
            os.getcwd() if cwd is None else os.path.join(os.getcwd(), _launch_text(cwd)),
            _launch_environment(arguments[3] if popen else arguments[2]),
        )
        engines = _launched_engines(
            [_launch_text(value) for value in argv], executable, context
        )
    if engines:
        raise RuntimeError(
            f"native_isolation.process.executable must exclude external engines; got {sorted(engines)!r}"
        )
    print(
        json.dumps({"native_process_launch": {"event": event, "executable": executable}}),
        file=sys.stderr,
        flush=True,
    )


def _refuse_qhull(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError(
        "native_isolation.geometry.operation must use a native owner; got a SciPy/Qhull geometry operation."
    )


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
    """Exercise format admission, CAD queries and CAD-derived native cells."""
    sys.meta_path.insert(0, _RefuseExternal())
    _block_qhull_operations()
    import numpy as np

    import phydrax as phx
    from phydrax.meshing import (
        load_meshcore,
        meshcore_identity,
        meshcore_runtime_identity,
    )

    # This route requires the own kernel; retain the loader's ABI evidence.
    library = load_meshcore()
    if not library.path.resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(
            f"native_isolation.library.path must resolve within the independent environment; got {library.path.resolve()}"
        )
    abi_contract = library["phx_mc_abi_contract"]()
    if not isinstance(abi_contract, bytes):
        raise RuntimeError(
            f"native_isolation.library.abi_contract must return contract bytes; got {type(abi_contract).__name__}"
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
            raise RuntimeError(
                f"native_isolation.cad.box_incidence must contain one solid and six faces; got format={format_name!r}, solids={decoded.topology.num_solids}, faces={decoded.topology.num_faces}"
            )
        containment = phx.geometry.prepare_brep_query(decoded).contains(
            np.asarray([[0.5, 1.0, 1.5], [2.0, 1.0, 1.5]], dtype=np.float64)
        )
        if np.asarray(containment.inside).tolist() != [True, False]:
            raise RuntimeError(
                f"native_isolation.cad.containment must classify the inside and outside probes; got format={format_name!r}, inside={np.asarray(containment.inside).tolist()}"
            )
        if np.any(
            np.asarray(containment.status) != phx.geometry.BRepProjectionStatus.UNIQUE
        ):
            raise RuntimeError(
                f"native_isolation.cad.containment_status must be unique; got format={format_name!r}, status={np.asarray(containment.status).tolist()}"
            )
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
            f"native_isolation.cad.restored_model_id must preserve the stored model; got restored={restored.model_id!r}, expected={decoded.model_id!r}"
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
        raise RuntimeError(
            f"native_isolation.generation.result must be a CellMeshingResult; got {type(generated).__name__}"
        )
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
            "native_isolation.generation.certification must be present; got None."
        )
    if not (
        generated.audit.passed
        and generated.compliance.passed
        and generated.certification.passed
    ):
        raise RuntimeError(
            f"native_isolation.generation.evidence must pass audit, compliance and certification; got {(generated.audit.passed, generated.compliance.passed, generated.certification.passed)!r}"
        )
    loaded = sorted(name for name in sys.modules if _external(name))
    if loaded:
        raise RuntimeError(
            f"native_isolation.import.loaded must exclude external engines; got {loaded!r}"
        )
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
            f"native_isolation.process.returncode must equal zero; got {result.returncode} for {' '.join(command)!r}"
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
    """Match installed package payloads to the resolver input wheels."""
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
                raise SystemExit(
                    f"native_isolation.wheel.payload must contain native package members; got none in {wheel.name!r}"
                )
            for name in members:
                package, relative = name.split("/", 1)
                spec = importlib.util.find_spec(package)
                if spec is None or spec.origin is None:
                    raise SystemExit(
                        f"native_isolation.wheel.package must be installed; got missing {package!r}"
                    )
                root = Path(spec.origin).resolve().parent
                if not root.is_relative_to(Path(sys.prefix).resolve()):
                    raise SystemExit(
                        f"native_isolation.wheel.package_path must resolve within the independent environment; got {package!r} outside the independent environment: {root}"
                    )
                installed = root / relative
                expected = hashlib.sha256(archive.read(name)).hexdigest()
                with installed.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != expected:
                    raise SystemExit(
                        f"native_isolation.wheel.payload must match the authenticated wheel; got Installed payload differs from {wheel.name}: {name}"
                    )
            evidence[wheel.name] = {"authenticated_payload_members": len(members)}
    print(
        json.dumps({"installed_wheel_authentication": evidence}, sort_keys=True),
        flush=True,
    )


def _copy_harness_inputs(
    source_root: Path, destination_root: Path, inputs: Sequence[str], /
) -> dict[str, str]:
    evidence = {}
    for relative in inputs:
        source = source_root / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        destination.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise RuntimeError(
                f"native_isolation.harness.payload must match source bytes; got {relative!r}"
            )
        destination.chmod(0o444)
        evidence[relative] = digest
    return evidence


def _stage_harness(site_packages: Path, /) -> dict[str, str]:
    """Stage audit drivers and their bounded input closure, not package code."""
    for namespace in ("tools", "examples"):
        shutil.copytree(
            ROOT / namespace,
            site_packages / namespace,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    # native-image imports neurofluid_qualification, whose sole benchmark
    # dependency is this numerical measurement helper (and no benchmark results).
    inputs = (
        "benchmarks/_runtime.py",
        "tests/data/cad/native_authored_boxes.json",
        "tests/data/meshing/corpus.json",
        "tests/data/meshing/curved_periodic_sources.json",
    )
    evidence = _copy_harness_inputs(ROOT, site_packages, inputs)
    print(
        json.dumps({"staged_harness_inputs_sha256": evidence}, sort_keys=True), flush=True
    )
    return evidence


def _stage_source_build_inputs(
    source_root: Path, site_packages: Path, /
) -> dict[str, str]:
    """Authenticate authored build provenance, not the installed dependency resolution.

    The wheel installs the package and, inside it, the native provider tree that
    ``native_provider_source_path`` reads. Those bytes are compared where runtime
    lookup reads them; a staged checkout copy would authenticate bytes nothing runs.
    """
    from benchmarks._runtime import source_build_fingerprint, source_build_inputs
    from phydrax.meshing.providers import native_provider_source_root

    source_root = source_root.resolve()
    # A malformed or dual installed provider layout refuses before anything is
    # staged beside the wheel-owned package.
    native_provider_source_root(site_packages / "phydrax")
    expected = source_build_fingerprint(source_root)
    providers = native_provider_source_root(source_root / "phydrax")
    # Installed package and provider bytes remain owned by the authenticated wheel.
    # Equality below checks their full inventory as well as their contents.
    inputs = [
        path.relative_to(source_root).as_posix()
        for path in source_build_inputs(source_root).values()
        if not path.is_relative_to(source_root / "phydrax")
        and (providers is None or not path.is_relative_to(providers))
    ]
    evidence = _copy_harness_inputs(source_root, site_packages, inputs)
    actual = source_build_fingerprint(site_packages)
    if actual != expected:
        raise SystemExit(
            "native_isolation.harness.source_build_identity must match the "
            "authored source/build input inventory; got installed package/input "
            f"fingerprint {actual!r}, expected {expected!r}"
        )
    print(
        json.dumps(
            {
                "authenticated_harness_source_build_fingerprint": actual,
                "staged_harness_source_build_inputs_sha256": evidence,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return evidence


def _fresh_distributed_checkpoint(workspace: Path, /) -> Path:
    """Reserve a fresh durable parent; the real producer creates its repository."""
    parent = Path(tempfile.mkdtemp(prefix="distributed-", dir=workspace))
    return parent / "checkpoint"


def _qualification(
    workspace: Path, arguments: Sequence[str], /, *, wheels: Sequence[Path] = ()
) -> None:
    """Execute the existing driver against installed packages, never checkout imports."""
    for name in ("phydrax", "phydrax_meshcore"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise SystemExit(
                f"native_isolation.qualification.package must be installed; got missing {name!r}"
            )
        if not Path(spec.origin).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise SystemExit(
                f"native_isolation.qualification.package_path must resolve within the independent environment; got {spec.origin!r}"
            )
    sys.meta_path.insert(0, _RefuseExternal())
    _block_qhull_operations()
    from tools._meshing_cases import (
        CORPUS,
        load_cad_fixture,
        load_corpus_registration,
        load_curved_periodic_source_registration,
        POLYHEDRAL_CASE_NAMES,
    )
    from tools.meshing_qualification._controls import (
        FAMILY_CASES,
        QualificationControls,
        SCENARIO_MODULES,
    )

    # Drivers are staged as an audit harness in this environment. No checkout
    # path or source package is inserted into the installed consumer's imports.
    sys.argv = ["meshing_qualification", *arguments]
    os.chdir(workspace)
    # Loaders authenticate reviewed records while retaining their
    # fingerprints, provenance and staged source-byte identities.
    load_cad_fixture()
    load_curved_periodic_source_registration()
    load_corpus_registration()
    namespace = runpy.run_module("tools.meshing_qualification")
    scenarios = SCENARIO_MODULES
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
        profiles = FAMILY_CASES
        commands.extend(
            ["--scenario", "native-family-lifecycle", "--family-case", name]
            for name in profiles
        )
        commands.extend(
            ["--scenario", "native-lifecycle", "--corpus-case", case.name]
            for case in CORPUS
            if case.expected_admission == "positive"
        )
        commands.extend(
            ["--scenario", "native-polyhedral-lifecycle", "--polyhedral-source", name]
            for name in POLYHEDRAL_CASE_NAMES
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
            if command == ["--scenario", "native-distributed-lifecycle"]:
                command.extend(
                    [
                        "--distributed-checkpoint-root",
                        str(_fresh_distributed_checkpoint(workspace)),
                    ]
                )
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
    scenario_module: Any = importlib.import_module(
        "tools.meshing_qualification." + scenarios[scenario]
    )
    original = scenario_module.run

    def require_execution(options: QualificationControls) -> dict[str, Any]:
        record = original(options)
        if record.get("status") != "passed":
            print(json.dumps(record, indent=2, sort_keys=True))
            raise SystemExit(
                f"native_isolation.qualification.status must equal passed; got {scenario}: qualification did not report passed execution ({record.get('status')!r})"
            )
        return record

    scenario_module.run = require_execution
    try:
        namespace["main"]()
    finally:
        scenario_module.run = original
    loaded = sorted(name for name in sys.modules if _external(name))
    if loaded:
        raise RuntimeError(
            f"native_isolation.qualification.loaded must exclude external engines; got {loaded!r}"
        )


def _consumer(workspace: Path, /) -> None:
    installed = _installed_external_modules()
    if installed:
        raise SystemExit(
            f"native_isolation.consumer.packages must exclude external engines and codecs; got {installed!r}"
        )
    for name in ("phydrax", "phydrax_meshcore"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise SystemExit(
                f"native_isolation.consumer.package must be installed; got missing {name!r}"
            )
        if not Path(spec.origin).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise SystemExit(
                f"native_isolation.consumer.package_path must resolve within the independent environment; got {spec.origin!r}"
            )
    print(json.dumps(smoke_native_workflows(workspace), indent=2, sort_keys=True))


def _missing_meshcore_consumer() -> None:
    """Prove neutral import and truthful refusal, not native construction."""
    sys.addaudithook(_refuse_external_process)
    if importlib.util.find_spec("phydrax_meshcore") is not None:
        raise SystemExit(
            "native_isolation.base.meshcore must be absent; got an installed meshcore package."
        )
    installed = _installed_external_modules()
    if installed:
        raise SystemExit(
            f"native_isolation.base.packages must exclude external engines and codecs; got {installed!r}"
        )
    sys.meta_path.insert(0, _RefuseExternal())
    import phydrax as phx
    from phydrax.meshing import (
        load_meshcore,
        meshcore_available,
        MeshcoreUnavailableError,
    )

    if not Path(phx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise SystemExit(
            f"native_isolation.base.package_path must resolve within the independent environment; got {phx.__file__!r}"
        )
    if meshcore_available():
        raise SystemExit(
            "native_isolation.base.meshcore_available must be false without the kernel; got True."
        )
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
            "native_isolation.base.load_meshcore must raise MeshcoreUnavailableError; got successful return without the kernel."
        )


def _isolation_startup(wheels: tuple[Path, ...]) -> None:
    """Install the same denial and wheel-authentication boundary in every child."""
    try:
        sys.meta_path.insert(0, _RefuseExternal())
        sys.addaudithook(_refuse_external_process)
        with contextlib.redirect_stdout(sys.stderr):
            _authenticate_wheels(wheels)
        _block_qhull_operations()
    except BaseException as error:
        print(f"Isolation startup refusal: {error}", file=sys.stderr, flush=True)
        os._exit(1)


def _dispatch_installed_consumer(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser, /
) -> bool:
    """Authenticate and execute child roles before parent installation work."""
    if arguments.isolation_startup is not None:
        _isolation_startup(tuple(arguments.isolation_startup))
        return True
    if arguments.stage_source_build_inputs is not None:
        if arguments.wheel is None or arguments.meshcore_wheel is None:
            parser.error(
                "native_isolation.source_build.wheels must include both input wheels; got a missing wheel argument"
            )
        _authenticate_wheels(
            (arguments.wheel.resolve(), arguments.meshcore_wheel.resolve())
        )
        _stage_source_build_inputs(arguments.stage_source_build_inputs.resolve(), ROOT)
        return True
    if arguments.consumer is not None:
        wheels = tuple(
            path.resolve()
            for path in (arguments.wheel, arguments.meshcore_wheel)
            if path is not None
        )
        if wheels:
            if len(wheels) != (1 if arguments.missing_meshcore_consumer else 2):
                raise SystemExit(
                    f"native_isolation.authentication.wheels must match the selected environment; got {len(wheels)} wheels"
                )
            _authenticate_wheels(wheels)
        if arguments.missing_meshcore_consumer:
            _missing_meshcore_consumer()
        elif arguments.qualification_consumer:
            command = json.loads(arguments.qualification[0])
            _qualification(arguments.consumer, command, wheels=wheels)
        else:
            _consumer(arguments.consumer)
        return True
    return False


def _prepare_installed_harness(
    python: Path,
    site_packages: Path,
    workspace: Path,
    wheel: Path,
    meshcore: Path,
    environment: dict[str, str],
) -> Path:
    """Install cold-child guards and authenticate the staged build inputs."""
    _stage_harness(site_packages)
    # The drivers spawn cold subprocesses; their denial boundary must be installed
    # before any staged helper or scientific consumer runs.
    (site_packages / "sitecustomize.py").write_text(
        "from tools.check_native_dependency_isolation import main\n"
        f"main(('--isolation-startup', {str(wheel)!r}, {str(meshcore)!r}))\n",
        encoding="utf-8",
    )
    runner = site_packages / "tools" / Path(__file__).name
    # Only installed dependencies execute this helper. The checkout supplies
    # authenticated read-only build inputs, never an import search path.
    _run(
        (
            str(python),
            "-I",
            str(runner),
            "--stage-source-build-inputs",
            str(ROOT),
            "--wheel",
            str(wheel),
            "--meshcore-wheel",
            str(meshcore),
        ),
        cwd=workspace,
        env=environment,
    )
    return runner


def _run_qualification_commands(
    arguments: argparse.Namespace,
    python: Path,
    runner: Path,
    consumer: Path,
    wheel: Path,
    meshcore: Path,
    environment: dict[str, str],
) -> None:
    """Admit native driver selections and execute them under installed guards."""
    if arguments.all_native:
        arguments.qualification.append(json.dumps(["--all-native"]))
    for encoded in arguments.qualification:
        command = json.loads(encoded)
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(value, str) for value in command)
        ):
            raise ValueError(
                f"native_isolation.qualification.arguments must be a nonempty JSON string array; got {command!r}"
            )
        if command != ["--all-native"]:
            if "--scenario" not in command or command.index("--scenario") + 1 >= len(
                command
            ):
                raise ValueError(
                    f"native_isolation.qualification.scenario must be explicitly selected; got {command!r}"
                )
            scenario = command[command.index("--scenario") + 1]
            if not scenario.startswith("native-"):
                raise ValueError(
                    f"native_isolation.qualification.scenario must name a native workflow; got {scenario!r}"
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--meshcore-wheel", type=Path)
    parser.add_argument("--python", default="3.12")
    parser.add_argument("--consumer", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--isolation-startup", type=Path, nargs=2, help=argparse.SUPPRESS)
    parser.add_argument("--stage-source-build-inputs", type=Path, help=argparse.SUPPRESS)
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
    arguments = parser.parse_args(argv)
    if _dispatch_installed_consumer(arguments, parser):
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
        runner = _prepare_installed_harness(
            python, site_packages, workspace, wheel, meshcore, environment
        )
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
        _run_qualification_commands(
            arguments, python, runner, consumer, wheel, meshcore, environment
        )


if __name__ == "__main__":
    main()
