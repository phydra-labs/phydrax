#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualify installed CUDA 13 wheels using an explicit interpreter and work root.

``--mode hardware`` requires positive ``--required-gpus`` and complete mandatory
cuDSS/AmgX/QUDA numerical and lifecycle coverage. Optional cuTENSOR executes its
full precision/derivative profile. Windows cuTensorNet/cuEST requests refuse.
The copied harness runs with ``-I`` against hash-bound installed wheel bytes.
Windows hostile audits require an existing user-local DLL directory supplied
through ``--hostile-library-path``; Linux retains its LD_LIBRARY_PATH audit.
Windows hardware audits inspect every loaded companion DLL and its installed,
RECORD-authenticated linker map through the shared PE policy. Binary hashes,
mapped dispatch markers and static-runtime exclusion accompany driver identity;
manifest declarations or matching version numbers alone never prove linkage.
``--mode installation-refusal`` exercises CPU selection, default-platform
no-GPU refusal and actual missing/corrupt installed files on Windows, and never
sets hardware_qualified. It restores injected installed-file faults after each
bounded child. Do not use that environment concurrently with other processes.
"""

from __future__ import annotations

import argparse
import email.policy
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from datetime import datetime, UTC
from email.parser import BytesParser
from pathlib import Path
from typing import TypedDict
from xml.etree import ElementTree

from tools._accelerator_process import CommandEvidence, run as run_bounded
from tools._cuda_optional_qualification import (
    HOSTILE_BOOTSTRAP,
    hostile_environment,
    LIBRARY_PROFILES,
    qualify_libraries,
    unavailable_libraries,
    validate_windows_runtime_evidence,
    WINDOWS_LOADER_PROVENANCE,
)


ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    "tests/unit/test_cuda_native_loader.py",
    "tests/integration/test_cuda_native_lifecycle.py",
    "tests/unit/linalg/test_cudss_provider.py",
    "tests/unit/backends/test_amg_backends.py",
    "tests/unit/backends/test_lattice_providers.py",
    "tests/unit/backends/test_backend_contracts.py",
)
# Test-runner requirements are taken from the wheel's own ``tests`` extra.
TEST_RUNNERS = ("pytest", "pytest-xdist", "hypothesis")
DEFAULT_HOSTILE_LIBRARY_PATH = "/usr/local/cuda/lib64"
DEFAULT_COMMAND_TIMEOUT = 1800.0
HOSTILE_LIBRARY_PATH_ENV = "PHYDRAX_CUDA_HOSTILE_LIBRARY_PATH"
DEVICE_COVERAGE_TESTS = (
    "test_cudss_same_plan_isolated_across_selected_devices[serial]",
    "test_cudss_same_plan_isolated_across_selected_devices[simultaneous]",
    "test_amgx_same_plan_isolated_across_selected_devices",
    "test_quda_independent_process_on_each_selected_device",
    *(
        f"test_exit_with_live_native_state_and_inflight_work[{environment}-{component}]"
        for environment in ("normal", "hostile-library-path")
        for component in ("cudss", "amgx", "quda")
    ),
)

AUDIT = (
    HOSTILE_BOOTSTRAP
    + WINDOWS_LOADER_PROVENANCE
    + """
import json, sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import phydrax as phx
from phydrax._cuda_native import CudaNativeUnavailableError, load_cuda_component
from phydrax._cuda_environment import competing_cuda_libraries
from phydrax.backends import cuda_environment

required_gpus = int(sys.argv[1])
hostile_library_path = sys.argv[2]
hostile_required = sys.argv[3] == "hostile"
try:
    components = {name: load_cuda_component(name) for name in ("cudss", "amgx", "quda")}
except CudaNativeUnavailableError as error:
    refusal = {"code": error.issue.code, "paths": list(error.issue.paths), "message": str(error)}
    print("AUDIT " + json.dumps({"refused": refusal}), flush=True)
    sys.exit(1)
devices = jax.devices("gpu")
if len(devices) < required_gpus:
    print("AUDIT " + json.dumps({
        "required_gpus": required_gpus, "visible_gpus": len(devices),
        "error": "Insufficient visible CUDA devices for hardware qualification",
    }), flush=True)
    sys.exit(1)
la = phx.linalg
size = 4
matrix = np.diag(np.arange(2.0, 2.0 + size)) + 0.1 * (np.ones((size, size)) - np.eye(size))
rows, columns = np.indices((size, size), dtype=np.int32)
relation = phx.sparse.EdgeRelation(
    columns.reshape(-1), rows.reshape(-1), source_size=size, target_size=size
)
operator = phx.sparse.SparseLinearMap(
    relation,
    jnp.asarray(matrix.reshape(-1), dtype=jnp.float32),
    operator_id="cuda-wheel-audit",
    properties=la.OperatorProperties(self_adjoint=True, evidence={"self_adjoint": "asserted"}),
)
prepared = la.prepare(la.LinearSystem(operator), la.LinearSolvePolicy(la.SparseLDLT(provider="cudss")))
solved = la.solve(prepared, jnp.ones(size, dtype=jnp.float32))
value = solved.value
residual = float(jnp.linalg.norm(jnp.asarray(matrix, dtype=jnp.float32) @ value - 1.0))
report = cuda_environment()
mapped = [
    {"soname": library.soname, "path": library.path, "major": library.major}
    for library in report.facts.mapped_libraries
]
owned = {os.path.normcase(str(Path(path).resolve())) for path, major in report.facts.wheel_library_majors if major == 13}
failures = [
    entry for entry in mapped
    if entry["major"] != 13 or os.path.normcase(str(Path(entry["path"]).resolve())) not in owned
]
runtime_identity = dict(report.facts.runtime_identity) if sys.platform == "win32" else None
loader_provenance = windows_loader_provenance() if sys.platform == "win32" else None
component_dispatch = windows_component_dispatch(loader_provenance) if sys.platform == "win32" else None
runtime_observed = sys.platform != "win32" or bool(runtime_identity and component_dispatch)
if sys.platform == "win32" and not set(components).issubset(component_dispatch):
    raise RuntimeError("Loaded mandatory components lack binary dispatch evidence")
hostile_candidates = competing_cuda_libraries(hostile_library_path, report.facts.mapped_libraries)
hostile_challenged = not hostile_required or bool(hostile_candidates)
print("AUDIT " + json.dumps({
    "residual": residual if np.isfinite(residual) else None,
    "solve_successful": bool(solved.successful),
    "solve_status": int(solved.status),
    "hostile_candidates": hostile_candidates,
    "hostile_challenged": hostile_challenged,
    "hostile_required": hostile_required,
    "successful_report": report.successful,
    "issues": [issue.code for issue in report.issues],
    "preloaded": {name: list(component.preloaded) for name, component in components.items()},
    "build_hashes": {name: component.build_hash for name, component in components.items()},
    "mapped": mapped,
    "failures": failures,
    "device": jax.devices()[0].device_kind,
    "compute_capability": jax.devices()[0].compute_capability,
    "driver_version": report.facts.driver_version,
    "required_gpus": required_gpus,
    "devices": [
        {"id": device.id, "kind": device.device_kind, "compute_capability": device.compute_capability}
        for device in devices
    ],
    "runtime_version": report.facts.runtime_version,
    "compiled_cuda_major": report.facts.compiled_cuda_major,
    "driver_compatibility_version": report.facts.driver_compatibility_version,
    "runtime_identity": runtime_identity,
    "loader_provenance": loader_provenance,
    "component_dispatch": component_dispatch,
    "jax": jax.__version__,
}), flush=True)
sys.exit(0 if report.successful and bool(solved.successful) and not failures and runtime_observed and residual < 1e-4 and hostile_challenged else 1)
"""
)


class Evidence(TypedDict):
    timestamp: str
    completed_at: str
    major: int
    required_gpus: int
    command_timeout_seconds: float
    selected_devices: list[int]
    libraries: list[str]
    library_qualification: dict[str, object] | None
    artifact_directory: str
    coverage: dict[str, list[int]]
    venv: str
    wheels: dict[str, str]
    install: list[CommandEvidence]
    tests: CommandEvidence | None
    audits: dict[str, object]
    extras_gate: CommandEvidence | None
    errors: list[str]
    successful: bool
    mode: str
    hardware_qualified: bool
    work_root: str
    harness: dict[str, str]
    identity: dict[str, object]
    installation_checks: list[CommandEvidence]


def run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    cwd: Path = ROOT,
    timeout: float = DEFAULT_COMMAND_TIMEOUT,
) -> CommandEvidence:
    return run_bounded(
        command,
        cwd=cwd,
        environment=dict(os.environ) if environment is None else environment,
        timeout=timeout,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_requirements(wheel: Path, *, required_extras: tuple[str, ...] = ()) -> list[str]:
    """Return the wheel's ``tests``-extra pins for the test-runner distributions."""
    with zipfile.ZipFile(wheel) as archive:
        member = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser(policy=email.policy.default).parsebytes(
            archive.read(member)
        )
    missing = set(required_extras) - set(metadata.get_all("Provides-Extra", []))
    if missing:
        raise ValueError(
            f"Main wheel does not declare requested extras: {sorted(missing)}"
        )
    requirements: list[str] = []
    for requirement in metadata.get_all("Requires-Dist", []):
        specifier, _, marker = requirement.partition(";")
        name = re.split(r"[<>=!~\[ ]", specifier.strip(), maxsplit=1)[0].lower()
        if name in TEST_RUNNERS and re.search(r"extra\s*==\s*['\"]tests['\"]", marker):
            requirements.append(specifier.strip())
    return requirements


def clean_environment(
    *, library_path: str | None, work_root: Path | None = None
) -> dict[str, str]:
    """Remove manual CUDA overrides and keep all child-owned caches local."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        not in (
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "CUDA_HOME",
            "CUDA_PATH",
            "CUDA_ROOT",
            "PYTHONPATH",
            "PYTHONHOME",
            "JAX_PLATFORMS",
            "JAX_PLATFORM_NAME",
            "XLA_FLAGS",
            "JAX_COMPILATION_CACHE_DIR",
            "QUDA_RESOURCE_PATH",
        )
        and not key.upper().startswith(("PHYDRAX_CUDA_", "CUDA_PATH_V", "UV_", "PIP_"))
    }
    environment.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if work_root is not None:
        for name, directory in (
            ("TEMP", "tmp"),
            ("TMP", "tmp"),
            ("TMPDIR", "tmp"),
            ("UV_CACHE_DIR", "uv-cache"),
            ("UV_PYTHON_INSTALL_DIR", "uv-python"),
            ("JAX_COMPILATION_CACHE_DIR", "jax-cache"),
            ("QUDA_RESOURCE_PATH", "quda-cache"),
            ("XDG_CACHE_HOME", "cache"),
            ("CUDA_CACHE_PATH", "cuda-cache"),
        ):
            path = validate_work_path(work_root / directory, work_root)
            path.mkdir(parents=True, exist_ok=True)
            environment[name] = str(path)
        environment["UV_PYTHON_DOWNLOADS"] = "never"
    return hostile_environment(environment, library_path)


def interpreter_path(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def validate_work_path(path: Path, root: Path) -> Path:
    """Reject links/reparse points before traversing any writable path component."""
    candidate = Path(os.path.abspath(path))
    boundary = Path(os.path.abspath(root))
    if not candidate.is_relative_to(boundary):
        raise ValueError(f"Writable path escapes work root: {path}")
    for part in (*reversed(candidate.parents), candidate):
        try:
            details = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode) or (
            getattr(details, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError(f"Writable path traverses a link/reparse point: {part}")
    return candidate


def copy_harness(destination: Path) -> dict[str, str]:
    """Copy only harness bytes; never expose the checkout's scientific package."""
    for directory in ("tests", "benchmarks", "tools"):
        source = ROOT / directory
        for path in source.rglob("*"):
            validate_work_path(path, ROOT)
        shutil.copytree(
            source,
            destination / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
    shutil.copy2(
        validate_work_path(ROOT / "pyproject.toml", ROOT), destination / "pyproject.toml"
    )
    return {
        path.relative_to(destination).as_posix(): sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }


INSTALLED_BOOTSTRAP = """
import sys
from pathlib import Path
import phydrax
if not Path(phydrax.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
    raise RuntimeError("Qualification did not import installed Phydrax bytes")
sys.path.insert(0, sys.argv[1])
"""


def installed_command(
    python: Path, harness: Path, module: str, args: Sequence[str]
) -> list[str]:
    return [
        str(python),
        "-I",
        "-c",
        INSTALLED_BOOTSTRAP
        + "\nimport runpy\nmodule = sys.argv[2]\nsys.argv = [module, *sys.argv[3:]]\nrunpy.run_module(module, run_name='__main__')",
        str(harness),
        module,
        *args,
    ]


IDENTITY = (
    WINDOWS_LOADER_PROVENANCE
    + """
import hashlib, importlib.metadata as metadata, json, sys
from pathlib import Path
expected = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
observed = {}
for name, record in expected.items():
    distribution = metadata.distribution(name)
    if distribution.version != record["version"]:
        raise RuntimeError(f"Installed {name} version differs from selected wheel")
    for member, digest in record["files"].items():
        path = Path(distribution.locate_file(member)).resolve()
        if not path.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f"Installed asset escapes environment: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Installed asset differs from selected wheel: {member}")
    observed[name] = record | {"installed_root": str(Path(distribution.locate_file("")).resolve())}
if sys.platform == "win32":
    from phydrax_cuda13_pjrt.runtime import installation_record
    if "jax" in sys.modules:
        raise RuntimeError("Metadata-only runtime import initialized JAX")
    runtime = installation_record()
    if "jax" in sys.modules:
        raise RuntimeError("Metadata-only runtime assessment initialized JAX")
    if runtime.issues or runtime.cuda_major != 13:
        raise RuntimeError(f"Invalid installed runtime: {runtime}")
    versions = {metadata.version(name) for name in ("phydrax", "phydrax-cuda13", "phydrax-cuda13-pjrt")}
    if len(versions) != 1 or runtime.version not in versions:
        raise RuntimeError("Core, companion and PJRT versions are not lockstep")
    if runtime.jax_version != "0.12.0" or runtime.jaxlib_version != "0.12.0":
        raise RuntimeError("Runtime requires the stock JAX/JAXlib 0.12.0 pair")
    if any(metadata.version(name) != "0.12.0" for name in ("jax", "jaxlib")):
        raise RuntimeError("Installed JAX/JAXlib pair differs from runtime")
    observed["runtime"] = runtime._asdict()
    observed["loader_provenance"] = windows_loader_provenance()
payload = json.dumps(observed, default=str, sort_keys=True).encode("utf-8")
Path(sys.argv[2]).write_bytes(payload)
print("IDENTITY " + hashlib.sha256(payload).hexdigest(), flush=True)
"""
)


def selected_identity(core: Path, wheelhouse: Path) -> dict[str, object]:
    wheels = [core]
    for name in (
        "phydrax_cuda13",
        *(("phydrax_cuda13_pjrt",) if sys.platform == "win32" else ()),
    ):
        candidates = [
            path
            for path in wheelhouse.glob(f"{name}-*.whl")
            if ("win_amd64" in path.name) == (sys.platform == "win32")
            and (
                sys.platform == "win32" or platform.machine().lower() in path.name.lower()
            )
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Require exactly one selected {name} wheel, got {candidates}"
            )
        wheels.extend(candidates)
    identity: dict[str, object] = {}
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            metadata_path = next(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            )
            metadata = BytesParser(policy=email.policy.default).parsebytes(
                archive.read(metadata_path)
            )
            identity[str(metadata["Name"])] = {
                "version": str(metadata["Version"]),
                "wheel_sha256": sha256(wheel),
                "files": {
                    name: hashlib.sha256(archive.read(name)).hexdigest()
                    for name in archive.namelist()
                    if not name.endswith("/") and not name.endswith(".dist-info/RECORD")
                },
            }
    return identity


def check_identity(
    python: Path, core: Path, wheelhouse: Path, evidence: Evidence
) -> None:
    try:
        expected = selected_identity(core, wheelhouse)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        evidence["errors"].append(f"Wheel identity refused: {error}")
        return
    root = Path(evidence["artifact_directory"])
    path = root / "selected-wheel-identity.json"
    observed = root / "installed-wheel-identity.json"
    path.write_text(json.dumps(expected, sort_keys=True), encoding="utf-8")
    result = run(
        [str(python), "-I", "-c", IDENTITY, str(path), str(observed)],
        cwd=root,
        environment=clean_environment(
            library_path=None, work_root=Path(evidence["work_root"])
        ),
        timeout=evidence["command_timeout_seconds"],
    )
    evidence["installation_checks"].append(result)
    records = [
        line[9:] for line in result["stdout"].splitlines() if line.startswith("IDENTITY ")
    ]
    if result["returncode"] or result["timed_out"] or len(records) != 1:
        evidence["errors"].append("Installed wheel/runtime identity binding failed")
        return
    try:
        if sha256(observed) != records[0]:
            raise ValueError("Installed identity sidecar hash mismatch")
        evidence["identity"] = json.loads(observed.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        evidence["errors"].append("Invalid installed identity evidence")


def install(
    python: Path, phydrax_wheel: Path, wheelhouse: Path, evidence: Evidence
) -> bool:
    extras = tuple(sorted(("cuda13", *evidence["libraries"])))
    try:
        runners = test_requirements(phydrax_wheel, required_extras=extras)
    except ValueError as error:
        evidence["errors"].append(str(error))
        return False
    references = sorted(
        {
            requirement
            for library in evidence["libraries"]
            for requirement in LIBRARY_PROFILES[library].extra_requirements
        }
    )
    command = [
        "uv",
        "pip",
        "install",
        # A consumer install never sees this repository's [tool.uv] settings.
        "--no-config",
        "--only-binary",
        ":all:",
        "--python",
        str(python),
        "--find-links",
        str(wheelhouse),
        "--reinstall-package",
        "phydrax",
        "--reinstall-package",
        "phydrax-cuda13",
        *(
            ["--reinstall-package", "phydrax-cuda13-pjrt"]
            if sys.platform == "win32"
            else []
        ),
        f"phydrax[{','.join(extras)}] @ {phydrax_wheel.as_uri()}",
        *runners,
        *references,
    ]
    result = run(
        command,
        timeout=evidence["command_timeout_seconds"],
        cwd=Path(evidence["artifact_directory"]),
        environment=clean_environment(
            library_path=None, work_root=Path(evidence["work_root"])
        ),
    )
    evidence["install"].append(result)
    if result["returncode"] or result["timed_out"]:
        evidence["errors"].append(f"Installation failed: {result['stderr']}")
        return False
    return True


def collect_test_coverage(path: Path, evidence: Evidence) -> None:
    """Reject absent, skipped, or partial hardware cases, even after pytest exits 0."""
    try:
        suite = ElementTree.parse(path)
    except (OSError, ElementTree.ParseError) as error:
        evidence["errors"].append(f"Missing or invalid provider test report: {error}")
        return
    required = [*DEVICE_COVERAGE_TESTS, "test_quda_refuses_multiple_processes"]
    if evidence["required_gpus"] >= 2:
        required.append("test_quda_refuses_second_device")
    expected = evidence["selected_devices"]
    if len(expected) != evidence["required_gpus"] or len(set(expected)) != len(expected):
        evidence["errors"].append("Selected CUDA topology does not match --required-gpus")
        return
    for name in required:
        cases = [case for case in suite.iter("testcase") if case.get("name") == name]
        if len(cases) != 1:
            evidence["errors"].append(
                f"Required native case missing or duplicated: {name}"
            )
            continue
        case = cases[0]
        if any(
            case.find(status) is not None for status in ("skipped", "failure", "error")
        ):
            evidence["errors"].append(f"Required native case did not pass: {name}")
            continue
        if name not in DEVICE_COVERAGE_TESTS:
            continue
        properties = [
            entry.get("value")
            for entry in case.findall("properties/property")
            if entry.get("name") == "cuda_native_devices"
        ]
        try:
            covered = (
                json.loads(properties[0])
                if len(properties) == 1 and properties[0] is not None
                else None
            )
        except (TypeError, json.JSONDecodeError):
            covered = None
        if (
            not isinstance(covered, list)
            or any(type(device) is not int for device in covered)
            or covered != expected
        ):
            evidence["errors"].append(
                f"Required device coverage missing for {name}: expected {expected}, got {covered}"
            )
            continue
        evidence["coverage"][name] = covered


def run_tests(
    python: Path,
    tests_root: Path,
    evidence: Evidence,
    *,
    hostile_library_path: str = DEFAULT_HOSTILE_LIBRARY_PATH,
) -> None:
    # Only the test package is importable from the tree; phydrax comes from the wheel.
    # One process owns the selected topology; the cuDSS race case owns its threads.
    # xdist workers must not concurrently initialize providers on the same GPU.
    environment = clean_environment(
        library_path=None, work_root=Path(evidence["work_root"])
    )
    environment["PHYDRAX_CUDA_QUALIFICATION_GPUS"] = str(evidence["required_gpus"])
    environment[HOSTILE_LIBRARY_PATH_ENV] = hostile_library_path
    report_path = tests_root / "native-provider-results.xml"
    result = run(
        [
            str(python),
            "-I",
            "-c",
            INSTALLED_BOOTSTRAP
            + "\nimport pytest\nraise SystemExit(pytest.main(sys.argv[2:]))",
            str(tests_root),
            "-p",
            "no:cacheprovider",
            "-q",
            "-rs",
            "-n",
            "0",
            "--junitxml",
            str(report_path),
            "-o",
            "junit_family=legacy",
            "--rootdir",
            str(tests_root),
            "-c",
            str(tests_root / "pyproject.toml"),
            *(str(tests_root / test) for test in TESTS),
        ],
        environment=environment,
        cwd=tests_root,
        timeout=evidence["command_timeout_seconds"],
    )
    evidence["tests"] = result
    if result["returncode"] or result["timed_out"]:
        evidence["errors"].append("Companion provider tests failed")
    collect_test_coverage(report_path, evidence)
    skipped = re.findall(r"^SKIPPED .*$", result["stdout"], flags=re.MULTILINE)
    # Every loader refusal text names the phydrax[cuda13] remedy.
    unavailable = [line for line in skipped if "phydrax[cuda13]" in line.lower()]
    if unavailable:
        evidence["errors"].append(
            f"Companion tests skipped as unavailable in a qualification run: {unavailable}"
        )
    if evidence["required_gpus"] >= 2:
        hardware_skips = [
            line
            for line in skipped
            if re.search(
                r"two (?:selected )?(?:CUDA devices|GPUs)", line, flags=re.IGNORECASE
            )
        ]
        if hardware_skips:
            evidence["errors"].append(
                f"Required two-device qualification skipped: {hardware_skips}"
            )


def audit(
    python: Path,
    label: str,
    library_path: str | None,
    evidence: Evidence,
) -> None:
    result = run(
        [
            str(python),
            "-I",
            "-c",
            AUDIT,
            str(evidence["required_gpus"]),
            library_path or "",
            label,
        ],
        environment=clean_environment(
            library_path=library_path, work_root=Path(evidence["work_root"])
        ),
        cwd=Path(evidence["artifact_directory"]),
        timeout=evidence["command_timeout_seconds"],
    )
    lines = [line for line in result["stdout"].splitlines() if line.startswith("AUDIT ")]
    try:
        report: object = (
            json.loads(lines[0].removeprefix("AUDIT ")) if len(lines) == 1 else None
        )
    except json.JSONDecodeError:
        report = None
    evidence["audits"][label] = {
        "library_path": library_path,
        "returncode": result["returncode"],
        "report": report,
        "stderr": result["stderr"][-4000:],
        "timed_out": result["timed_out"],
        "command": result,
    }
    if isinstance(report, dict) and report.get("refused") is not None:
        evidence["errors"].append(
            f"Mapped-library audit {label!r} refused: "
            + json.dumps(report["refused"], sort_keys=True)
        )
        return
    if sys.platform == "win32" and isinstance(report, dict):
        try:
            validate_windows_runtime_evidence(report)
            if report["loader_provenance"] != evidence["identity"].get(
                "loader_provenance"
            ):
                raise ValueError(
                    "Loader provenance changed after installed-wheel identity check"
                )
            clean = evidence["audits"].get("clean")
            if label != "clean" and isinstance(clean, dict):
                clean_report = clean.get("report")
                if not isinstance(clean_report, dict) or any(
                    clean_report.get(key) != report[key]
                    for key in ("runtime_identity", "component_dispatch")
                ):
                    raise ValueError(
                        "Hostile audit changed the Windows driver cohort or component dispatch"
                    )
        except ValueError as error:
            evidence["errors"].append(f"Mapped-library audit {label!r}: {error}")
    if result["returncode"] or result["timed_out"] or report is None:
        evidence["errors"].append(f"Mapped-library audit {label!r} failed")
    elif label == "clean" and isinstance(report, dict):
        evidence["selected_devices"] = [
            device["id"] for device in report["devices"][: evidence["required_gpus"]]
        ]


def installation_refusal_checks(python: Path, evidence: Evidence) -> None:
    """CPU import and hidden-device refusal are not numerical qualification."""
    root = Path(evidence["artifact_directory"])
    checks = (
        (
            "cpu",
            """
import sys
from pathlib import Path
import phydrax, jax
assert Path(phydrax.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert all(device.platform == "cpu" for device in jax.devices())
if sys.platform == "win32":
    from phydrax_cuda13_pjrt.plugin import bootstrap_state
    state = bootstrap_state()
    assert state.status in ("uninitialized", "cpu_selected"), state
print("INSTALLATION_CHECK cpu passed", flush=True)
""",
        ),
        *(
            (
                platform,
                """
try:
    import phydrax, jax
    devices = jax.devices()
except (RuntimeError, ImportError) as error:
    if not any(word in str(error).lower() for word in ("cuda", "gpu", "nvidia")):
        raise
    print("INSTALLATION_CHECK hidden CUDA refused: " + str(error), flush=True)
else:
    raise RuntimeError(f"Hidden-device CUDA request did not refuse: {devices}")
""",
            )
            for platform in ("", "cuda")
        ),
        (
            "",
            """
import jax
from phydrax_cuda13_pjrt import plugin
original = OSError("qualification injected native bootstrap failure")
def fail_native(bridge):
    raise original
plugin._register_native = fail_native
try:
    plugin.initialize()
except RuntimeError as error:
    assert error.__cause__ is original
else:
    raise AssertionError("Injected native bootstrap unexpectedly succeeded")
assert plugin.bootstrap_state().status == "failed"
try:
    jax.devices()
except RuntimeError as error:
    assert str(original) in str(error), str(error)
    print("INSTALLATION_CHECK injected bootstrap failure refused", flush=True)
else:
    raise AssertionError("JAX selected CPU after injected CUDA bootstrap failure")
""",
        ),
    )
    for selected_platform, source in checks:
        environment = clean_environment(
            library_path=None, work_root=Path(evidence["work_root"])
        )
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
        if selected_platform:
            environment["JAX_PLATFORMS"] = selected_platform
        result = run(
            [str(python), "-I", "-c", source],
            cwd=root,
            environment=environment,
            timeout=evidence["command_timeout_seconds"],
        )
        evidence["installation_checks"].append(result)
        if (
            result["returncode"]
            or result["timed_out"]
            or "INSTALLATION_CHECK " not in result["stdout"]
        ):
            evidence["errors"].append(
                f"Installation/refusal {selected_platform} check failed"
            )
    if not evidence["errors"]:
        installed_fault_checks(python, evidence)


def installed_fault_checks(python: Path, evidence: Evidence) -> None:
    """Exercise real installed-file failures; restore bytes after each bounded child."""
    identity = evidence["identity"]
    runtime = identity.get("runtime")
    if not isinstance(runtime, dict):
        evidence["errors"].append("Missing runtime identity for installed fault checks")
        return
    root = Path(evidence["work_root"])
    manifest = validate_work_path(Path(str(runtime["manifest_path"])), root)
    faults: list[tuple[str, Path, bytes | None]] = [
        ("missing-manifest", manifest, None),
        ("corrupt-manifest", manifest, b"{"),
    ]
    for distribution in ("phydrax-cuda13", "phydrax-cuda13-pjrt"):
        record = identity.get(distribution)
        if not isinstance(record, dict) or not isinstance(record.get("files"), dict):
            evidence["errors"].append(f"Missing identity for {distribution} refusal")
            return
        member = next(
            name for name in record["files"] if name.endswith(".dist-info/METADATA")
        )
        path = validate_work_path(Path(str(record["installed_root"])) / member, root)
        faults.append((f"missing-{distribution}-metadata", path, None))
        original = path.read_bytes()
        version = str(record["version"]).encode()
        faults.append(
            (
                f"mismatched-{distribution}-version",
                path,
                original.replace(b"Version: " + version, b"Version: 0.0.0"),
            )
        )
    assets = runtime.get("assets")
    if not isinstance(assets, list):
        evidence["errors"].append("Missing runtime assets for installed fault checks")
        return
    pjrt = next(
        (
            asset
            for asset in assets
            if isinstance(asset, list) and len(asset) == 4 and asset[3] == "pjrt"
        ),
        None,
    )
    if pjrt is None:
        evidence["errors"].append("Missing PJRT asset identity")
        return
    faults.append(("missing-pjrt", validate_work_path(Path(pjrt[0]), root), None))
    faults.append(
        ("corrupt-pjrt", validate_work_path(Path(pjrt[0]), root), b"invalid PE")
    )
    script = """
import json, sys
try:
    import phydrax, jax
    devices = jax.devices()
except (RuntimeError, ImportError, OSError, ValueError) as error:
    if not any(word in str(error).lower() for word in json.loads(sys.argv[1])):
        raise
    print("INSTALLATION_FAULT_REFUSED " + repr(error), flush=True)
else:
    raise RuntimeError(f"Corrupt installed CUDA intent fell back or succeeded: {devices}")
"""
    for label, path, replacement in faults:
        backup = path.with_name(path.name + ".qualification-backup")
        validate_work_path(backup, root)
        if backup.exists():
            raise FileExistsError(f"Refuse overwriting existing backup: {backup}")
        path.rename(backup)
        try:
            if replacement is not None:
                path.write_bytes(replacement)
            environment = clean_environment(library_path=None, work_root=root)
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
            reasons = (
                ["version", "0.0.0"]
                if label.endswith("-version")
                else [
                    "metadata",
                    label.removeprefix("missing-").removesuffix("-metadata"),
                ]
                if label.endswith("-metadata")
                else [path.name.lower(), "manifest"]
            )
            result = run(
                [str(python), "-I", "-c", script, json.dumps(reasons)],
                cwd=Path(evidence["artifact_directory"]),
                environment=environment,
                timeout=evidence["command_timeout_seconds"],
            )
            evidence["installation_checks"].append(result)
            if (
                result["returncode"]
                or result["timed_out"]
                or "INSTALLATION_FAULT_REFUSED " not in result["stdout"]
            ):
                evidence["errors"].append(
                    f"Installed {label} did not refuse with default platforms"
                )
        finally:
            if path.exists():
                path.unlink()
            backup.rename(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True, help="Qualification venv")
    parser.add_argument(
        "--python", type=Path, required=True, help="Explicit CPython 3.13 interpreter"
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        required=True,
        help="Root for all writable qualification artifacts",
    )
    parser.add_argument(
        "--mode", choices=("hardware", "installation-refusal"), default="hardware"
    )
    parser.add_argument(
        "--required-gpus",
        type=int,
        default=1,
        help="Exercise the first N visible CUDA devices in every required provider/lifecycle case.",
    )
    parser.add_argument(
        "--libraries",
        nargs="*",
        choices=tuple(LIBRARY_PROFILES),
        default=[],
        help="Optional numerical providers; full local mathematics runs on every selected GPU.",
    )
    parser.add_argument(
        "--reuse-venv",
        action="store_true",
        help="Install into an existing venv instead of creating a fresh one",
    )
    parser.add_argument("--phydrax-wheel", type=Path, required=True)
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        required=True,
        help="Directory with the companion and phydrax-cuda-major wheels",
    )
    parser.add_argument(
        "--hostile-library-path",
        default=None,
        help="Existing hostile CUDA library directory; Windows requires a work-root-local DLL directory",
    )
    parser.add_argument(
        "--extras-gate-work-dir",
        type=Path,
        default=None,
        help="Also run tools/check_cuda_extras.py with this scratch directory",
    )
    parser.add_argument("--linux-tag", default="manylinux_2_28")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_COMMAND_TIMEOUT,
        help="Whole-process-group deadline in seconds for each command.",
    )
    args = parser.parse_args()
    if args.required_gpus < 1:
        parser.error("--required-gpus must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if len(args.libraries) != len(set(args.libraries)):
        parser.error("--libraries must not repeat a provider")
    try:
        work_root = validate_work_path(args.work_root, args.work_root)
        if sys.platform == "win32":
            validate_work_path(work_root, Path.home())
        for path in (args.venv, args.evidence, args.extras_gate_work_dir):
            if path is not None:
                validate_work_path(path, work_root)
    except ValueError as error:
        parser.error(str(error))
    if not args.python.is_file():
        parser.error("--python must name an existing interpreter")
    if args.mode == "installation-refusal" and sys.platform != "win32":
        parser.error("installation-refusal mode is the Windows no-GPU gate")
    if args.mode == "hardware":
        if sys.platform == "win32":
            if args.hostile_library_path is None:
                parser.error("Windows hardware mode requires --hostile-library-path")
            try:
                hostile = validate_work_path(Path(args.hostile_library_path), work_root)
            except ValueError as error:
                parser.error(str(error))
            if not hostile.is_relative_to(work_root) or not hostile.is_dir():
                parser.error("Windows hostile DLL directory must exist under --work-root")
            try:
                for path in hostile.rglob("*"):
                    validate_work_path(path, work_root)
            except ValueError as error:
                parser.error(str(error))
            args.hostile_library_path = str(hostile)
        else:
            args.hostile_library_path = (
                args.hostile_library_path or DEFAULT_HOSTILE_LIBRARY_PATH
            )
    work_root.mkdir(parents=True, exist_ok=True)
    venv: Path = args.venv.resolve()
    phydrax_wheel: Path = args.phydrax_wheel.resolve()
    wheelhouse: Path = args.wheelhouse.resolve()
    evidence: Evidence = {
        "timestamp": datetime.now(UTC).isoformat(),
        "completed_at": "",
        "major": 13,
        "required_gpus": args.required_gpus,
        "command_timeout_seconds": args.timeout,
        "selected_devices": [],
        "libraries": sorted(args.libraries),
        "library_qualification": None,
        "artifact_directory": str(
            Path(tempfile.mkdtemp(prefix="wheel-qualification-", dir=work_root))
        ),
        "coverage": {},
        "venv": str(venv),
        "wheels": {
            path.name: sha256(path)
            for path in (*sorted(wheelhouse.glob("*.whl")), phydrax_wheel)
        },
        "install": [],
        "tests": None,
        "audits": {},
        "extras_gate": None,
        "errors": [],
        "successful": False,
        "mode": args.mode,
        "hardware_qualified": False,
        "work_root": str(work_root),
        "harness": {},
        "identity": {},
        "installation_checks": [],
    }
    python = interpreter_path(venv)
    unavailable = unavailable_libraries(tuple(evidence["libraries"]))
    if unavailable:
        evidence["library_qualification"] = {
            "refused": list(unavailable),
            "hardware_qualified": False,
        }
        evidence["errors"].extend(
            f"{library} is unavailable on Windows; explicit request refused, not numerical qualification"
            for library in unavailable
        )
    if not evidence["errors"] and not args.reuse_venv:
        if venv.exists():
            evidence["errors"].append(f"{venv} exists; pass --reuse-venv or remove it")
        else:
            created = run(
                [
                    "uv",
                    "--no-config",
                    "venv",
                    "--python",
                    str(args.python.resolve()),
                    str(venv),
                ],
                timeout=args.timeout,
                cwd=work_root,
                environment=clean_environment(library_path=None, work_root=work_root),
            )
            evidence["install"].append(created)
            if created["returncode"] or created["timed_out"]:
                evidence["errors"].append(f"venv creation failed: {created['stderr']}")
    if not evidence["errors"] and install(python, phydrax_wheel, wheelhouse, evidence):
        check_identity(python, phydrax_wheel, wheelhouse, evidence)
    if not evidence["errors"] and args.mode == "installation-refusal":
        installation_refusal_checks(python, evidence)
    if not evidence["errors"] and args.mode == "hardware":
        # Gate hardware availability before tests can skip.
        audit(python, "clean", None, evidence)
    if not evidence["errors"] and args.mode == "hardware":
        evidence_path: Path = args.evidence.resolve()
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        tests_root = Path(evidence["artifact_directory"])
        evidence["harness"] = copy_harness(tests_root)
        run_tests(
            python,
            tests_root,
            evidence,
            hostile_library_path=args.hostile_library_path,
        )
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not evidence["errors"] and evidence["libraries"]:
            optional, failures = qualify_libraries(
                python,
                tests_root,
                tuple(evidence["libraries"]),
                evidence["required_gpus"],
                clean_environment(library_path=None, work_root=work_root),
                evidence["command_timeout_seconds"],
                args.hostile_library_path,
            )
            evidence["library_qualification"] = optional
            evidence["errors"].extend(failures)
        if not evidence["errors"]:
            audit(python, "hostile", args.hostile_library_path, evidence)
        if not evidence["errors"] and args.extras_gate_work_dir is not None:
            markers = list(wheelhouse.glob("phydrax_cuda_major-13*.whl"))
            if len(markers) != 1:
                evidence["errors"].append(
                    "Extras gate requires exactly one CUDA 13 marker wheel in the wheelhouse"
                )
        if not evidence["errors"] and args.extras_gate_work_dir is not None:
            gate = run(
                [
                    str(args.python.resolve()),
                    "-m",
                    "tools.check_cuda_extras",
                    "--python",
                    str(args.python.resolve()),
                    "--phydrax-wheel",
                    str(phydrax_wheel),
                    "--marker-wheel",
                    str(markers[0]),
                    "--work-dir",
                    str(args.extras_gate_work_dir.resolve()),
                    "--evidence",
                    str(args.extras_gate_work_dir.resolve() / "extras-evidence.json"),
                    "--companion-wheelhouse",
                    str(wheelhouse),
                    "--linux-tag",
                    args.linux_tag,
                ],
                timeout=args.timeout,
                cwd=tests_root,
                environment=clean_environment(library_path=None, work_root=work_root),
            )
            evidence["extras_gate"] = gate
            if gate["returncode"] or gate["timed_out"]:
                evidence["errors"].append("CUDA extras gate failed")
    evidence["successful"] = not evidence["errors"]
    evidence["hardware_qualified"] = evidence["successful"] and args.mode == "hardware"
    evidence["completed_at"] = datetime.now(UTC).isoformat()
    evidence_path: Path = args.evidence.resolve()
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for error in evidence["errors"]:
        print(error, file=sys.stderr)
    print(f"Evidence: {evidence_path}")
    return 0 if evidence["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
