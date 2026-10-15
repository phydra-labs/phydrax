#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualify a CUDA companion release in a clean environment on a real GPU.

The tool installs ``phydrax[cudaNN]`` from a built Phydrax wheel plus a wheelhouse
holding the companion and CUDA-major marker wheels (until they are published),
runs the companion provider tests against the installed packages, audits the
CUDA libraries mapped by a process that loaded every component (normally and
under a hostile ``LD_LIBRARY_PATH``), and writes JSON evidence.
"""

from __future__ import annotations

import argparse
import email.policy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, UTC
from email.parser import BytesParser
from pathlib import Path
from typing import TypedDict


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

AUDIT = """
import json, os, sys
import jax
import jax.numpy as jnp
import numpy as np
import phydrax as phx
from phydrax._cuda_native import CudaNativeUnavailableError, load_cuda_component
from phydrax.backends import cuda_environment

major = int(sys.argv[1])
hostile = sys.argv[2] if len(sys.argv) > 2 else ""
try:
    components = {name: load_cuda_component(name) for name in ("cudss", "amgx", "quda")}
except CudaNativeUnavailableError as error:
    # JAX's CUDA 12 plugin may fail under a shadowing path; the refusal must name it.
    refusal = {"code": error.issue.code, "paths": list(error.issue.paths), "message": str(error)}
    accepted = (
        major == 12 and hostile in error.issue.paths
        and error.issue.code == "backend-unavailable"
    )
    print("AUDIT " + json.dumps({"refused": refusal, "accepted_refusal": accepted}), flush=True)
    sys.exit(0 if accepted else 1)
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
value = la.solve(prepared, jnp.ones(size, dtype=jnp.float32)).value
residual = float(jnp.linalg.norm(jnp.asarray(matrix, dtype=jnp.float32) @ value - 1.0))
report = cuda_environment()
mapped = [
    {"soname": library.soname, "path": library.path, "major": library.major}
    for library in report.facts.mapped_libraries
]
failures = [
    entry for entry in mapped
    if entry["major"] != major or entry["path"].startswith("/usr/local/cuda")
]
print("AUDIT " + json.dumps({
    "residual": residual,
    "successful_report": report.successful,
    "issues": [issue.code for issue in report.issues],
    "preloaded": {name: list(component.preloaded) for name, component in components.items()},
    "build_hashes": {name: component.build_hash for name, component in components.items()},
    "mapped": mapped,
    "failures": failures,
    "device": jax.devices()[0].device_kind,
    "compute_capability": jax.devices()[0].compute_capability,
    "driver_version": report.facts.driver_version,
    "runtime_version": report.facts.runtime_version,
    "jax": jax.__version__,
}), flush=True)
sys.exit(0 if not failures and residual < 1e-4 else 1)
"""


class CommandEvidence(TypedDict):
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


class Evidence(TypedDict):
    timestamp: str
    completed_at: str
    major: int
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
) -> CommandEvidence:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_requirements(wheel: Path) -> list[str]:
    """Return the wheel's ``tests``-extra pins for the test-runner distributions."""
    with zipfile.ZipFile(wheel) as archive:
        member = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser(policy=email.policy.default).parsebytes(
            archive.read(member)
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
    python: Path, phydrax_wheel: Path, wheelhouse: Path, major: int, evidence: Evidence
) -> bool:
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
        f"phydrax-cuda{major}",
        f"phydrax[cuda{major}] @ {phydrax_wheel.as_uri()}",
        *test_requirements(phydrax_wheel),
    ]
    result = run(command)
    evidence["install"].append(result)
    if result["returncode"]:
        evidence["errors"].append(f"Installation failed: {result['stderr']}")
        return False
    return True


def run_tests(python: Path, tests_root: Path, evidence: Evidence) -> None:
    # Only the test package is importable from the tree; phydrax comes from the wheel.
    environment = clean_environment(library_path=None)
    environment["PYTHONPATH"] = str(tests_root)
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
            "2",
            "--rootdir",
            str(ROOT),
            "-c",
            str(ROOT / "pyproject.toml"),
            *(str(ROOT / test) for test in TESTS),
        ],
        environment=environment,
        cwd=tests_root,
    )
    evidence["tests"] = result
    if result["returncode"]:
        evidence["errors"].append("Companion provider tests failed")
    skipped = re.findall(r"^SKIPPED .*$", result["stdout"], flags=re.MULTILINE)
    # Every loader refusal text names the phydrax[cudaNN] remedy.
    unavailable = [line for line in skipped if "phydrax[cuda" in line.lower()]
    if unavailable:
        evidence["errors"].append(
            f"Companion tests skipped as unavailable in a qualification run: {unavailable}"
        )


def audit(
    python: Path, major: int, label: str, library_path: str | None, evidence: Evidence
) -> None:
    result = run(
        [str(python), "-c", AUDIT, str(major), library_path or ""],
        environment=clean_environment(library_path=library_path),
        cwd=Path(tempfile.gettempdir()),
    )
    lines = [line for line in result["stdout"].splitlines() if line.startswith("AUDIT ")]
    report: object = json.loads(lines[-1].removeprefix("AUDIT ")) if lines else None
    evidence["audits"][label] = {
        "library_path": library_path,
        "returncode": result["returncode"],
        "report": report,
        "stderr": result["stderr"][-4000:],
    }
    if result["returncode"] or report is None:
        evidence["errors"].append(f"Mapped-library audit {label!r} failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--major", type=int, choices=(12, 13), required=True)
    parser.add_argument("--venv", type=Path, required=True, help="Qualification venv")
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
    args = parser.parse_args()
    major: int = args.major
    venv: Path = args.venv.resolve()
    phydrax_wheel: Path = args.phydrax_wheel.resolve()
    wheelhouse: Path = args.wheelhouse.resolve()
    evidence: Evidence = {
        "timestamp": datetime.now(UTC).isoformat(),
        "completed_at": "",
        "major": major,
        "venv": str(venv),
        "wheels": {
            path.name: sha256(path)
            for path in (phydrax_wheel, *sorted(wheelhouse.glob("*.whl")))
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
            created = run(["uv", "venv", "--python", "3.12", str(venv)])
            evidence["install"].append(created)
            if created["returncode"]:
                evidence["errors"].append(f"venv creation failed: {created['stderr']}")
    if not evidence["errors"] and install(
        python, phydrax_wheel, wheelhouse, major, evidence
    ):
        with tempfile.TemporaryDirectory(prefix="phydrax-cuda-qualify-") as scratch:
            tests_root = Path(scratch)
            (tests_root / "tests").symlink_to(ROOT / "tests", target_is_directory=True)
            run_tests(python, tests_root, evidence)
        audit(python, major, "clean", None, evidence)
        audit(python, major, "hostile", args.hostile_library_path, evidence)
        if args.extras_gate_work_dir is not None:
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
                ]
            )
            evidence["extras_gate"] = gate
            if gate["returncode"]:
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
