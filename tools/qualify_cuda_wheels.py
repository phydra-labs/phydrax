#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualify the CUDA 13 companion release in a clean environment on a real GPU.

The tool installs ``phydrax[cuda13]`` from a built Phydrax wheel plus a wheelhouse
holding the companion and CUDA-major marker wheels (until they are published),
runs the companion provider tests against the installed packages on exactly the
first ``--required-gpus`` visible devices, audits the CUDA libraries mapped by a
process that loaded every component (normally and under a hostile
``LD_LIBRARY_PATH``), and writes JSON evidence. Required provider and teardown
cases must pass and report every selected device; visibility alone is insufficient.
"""

from __future__ import annotations

import argparse
import email.policy
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import zipfile
from datetime import datetime, UTC
from email.parser import BytesParser
from pathlib import Path
from typing import TypedDict
from xml.etree import ElementTree

from tools._accelerator_process import CommandEvidence, run as run_bounded
from tools._cuda_optional_qualification import LIBRARY_PROFILES, qualify_libraries


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

AUDIT = """
import json, sys
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
failures = [
    entry for entry in mapped
    if entry["major"] != 13 or entry["path"].startswith("/usr/local/cuda")
]
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
    "jax": jax.__version__,
}), flush=True)
sys.exit(0 if report.successful and bool(solved.successful) and not failures and residual < 1e-4 and hostile_challenged else 1)
"""


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


def clean_environment(*, library_path: str | None) -> dict[str, str]:
    """Process environment without CUDA toolkit overrides or developer library paths."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in (
            "LD_LIBRARY_PATH",
            "CUDA_HOME",
            "CUDA_PATH",
            "CUDA_ROOT",
            "PYTHONPATH",
            "JAX_PLATFORMS",
        )
        and not key.startswith("PHYDRAX_CUDA_")
    }
    environment.update(
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    if library_path is not None:
        environment["LD_LIBRARY_PATH"] = library_path
    return environment


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
        "--python",
        str(python),
        "--find-links",
        str(wheelhouse),
        "--reinstall-package",
        "phydrax",
        "--reinstall-package",
        "phydrax-cuda13",
        f"phydrax[{','.join(extras)}] @ {phydrax_wheel.as_uri()}",
        *runners,
        *references,
    ]
    result = run(command, timeout=evidence["command_timeout_seconds"])
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
    environment = clean_environment(library_path=None)
    environment["PYTHONPATH"] = str(tests_root)
    environment["PHYDRAX_CUDA_QUALIFICATION_GPUS"] = str(evidence["required_gpus"])
    environment[HOSTILE_LIBRARY_PATH_ENV] = hostile_library_path
    report_path = tests_root / "native-provider-results.xml"
    result = run(
        [
            str(python),
            "-m",
            "pytest",
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
            str(ROOT),
            "-c",
            str(ROOT / "pyproject.toml"),
            *(str(ROOT / test) for test in TESTS),
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
            "-c",
            AUDIT,
            str(evidence["required_gpus"]),
            library_path or "",
            label,
        ],
        environment=clean_environment(library_path=library_path),
        cwd=Path(tempfile.gettempdir()),
        timeout=evidence["command_timeout_seconds"],
    )
    lines = [line for line in result["stdout"].splitlines() if line.startswith("AUDIT ")]
    report: object = json.loads(lines[-1].removeprefix("AUDIT ")) if lines else None
    evidence["audits"][label] = {
        "library_path": library_path,
        "returncode": result["returncode"],
        "report": report,
        "stderr": result["stderr"][-4000:],
        "timed_out": result["timed_out"],
        "command": result,
    }
    if result["returncode"] or result["timed_out"] or report is None:
        evidence["errors"].append(f"Mapped-library audit {label!r} failed")
    elif label == "clean" and isinstance(report, dict):
        evidence["selected_devices"] = [
            device["id"] for device in report["devices"][: evidence["required_gpus"]]
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True, help="Qualification venv")
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
        default=DEFAULT_HOSTILE_LIBRARY_PATH,
        help="LD_LIBRARY_PATH pointing at a system CUDA toolkit for the hostile audit",
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
        "artifact_directory": "",
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
    }
    python = venv / "bin" / "python"
    if not args.reuse_venv:
        if venv.exists():
            evidence["errors"].append(f"{venv} exists; pass --reuse-venv or remove it")
        else:
            created = run(
                ["uv", "venv", "--python", "3.13", str(venv)], timeout=args.timeout
            )
            evidence["install"].append(created)
            if created["returncode"] or created["timed_out"]:
                evidence["errors"].append(f"venv creation failed: {created['stderr']}")
    if not evidence["errors"] and install(python, phydrax_wheel, wheelhouse, evidence):
        # Gate hardware availability before tests can skip.
        audit(python, "clean", None, evidence)
    if not evidence["errors"]:
        evidence_path: Path = args.evidence.resolve()
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        tests_root = Path(
            tempfile.mkdtemp(
                prefix=evidence_path.stem + "-artifacts-", dir=evidence_path.parent
            )
        )
        evidence["artifact_directory"] = str(tests_root)
        for directory in ("tests", "benchmarks", "tools"):
            (tests_root / directory).symlink_to(
                ROOT / directory, target_is_directory=True
            )
        (tests_root / "pyproject.toml").symlink_to(ROOT / "pyproject.toml")
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
                clean_environment(library_path=None),
                evidence["command_timeout_seconds"],
                args.hostile_library_path,
            )
            evidence["library_qualification"] = optional
            evidence["errors"].extend(failures)
        if not evidence["errors"]:
            audit(python, "hostile", args.hostile_library_path, evidence)
        if not evidence["errors"] and args.extras_gate_work_dir is not None:
            gate = run(
                [
                    sys.executable,
                    str(ROOT / "tools/check_cuda_extras.py"),
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
            )
            evidence["extras_gate"] = gate
            if gate["returncode"] or gate["timed_out"]:
                evidence["errors"].append("CUDA extras gate failed")
    evidence["successful"] = not evidence["errors"]
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
