#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host-only DAFoam 4.0.3 steady MPhys profile with same-realization totals.

This adapter launches our bundled worker using a pinned Python executable. The
worker invokes PYDAFOAM, DAFoamSolver, DAFoamFunctions and OpenMDAO compute_totals;
its JSON is transport produced by PHYDRAX, not an alleged DAFoam file format.
No native PHYDRAX residual or differentiation through a host process is exposed.
The reference APIs are pinned to https://github.com/mdolab/dafoam/tree/v4.0.3 .
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .._fingerprint import canonical_fingerprint, canonical_json
from ..artifacts import ScientificArtifactEnvelope
from ._energy_worker import _digest_file, _relative_path
from .energy_runtime import (
    _artifact,
    _host_only,
    EnergyRunResult,
    PinnedExecutable,
    run_energy_command,
)


@dataclass(frozen=True, slots=True)
class DAFoamRuntime:
    """Exact Python/native implementation file pins, rechecked in the worker.

    package_root is the directory containing dafoam/pyDAFoam.py, i.e. the
    *package* directory, not its parent. Extra runtime roots should include the
    OpenFOAM/DAFoam libraries and the installed OpenMDAO, MPhys, PETSc, MPI and
    IDWarp implementations. System libraries outside these roots are not pinned.
    This is execution provenance and operational isolation, not a sandbox.
    """

    python: PinnedExecutable
    package_root: str
    implementation_files: tuple[tuple[str, str], ...]
    license_id: str
    version: str = "4.0.3"

    def __post_init__(self) -> None:
        _host_only()
        if self.version != "4.0.3":
            raise ValueError("The DAFoam MPhys profile supports exactly 4.0.3.")
        if not self.license_id.strip():
            raise ValueError("An explicit DAFoam license identifier is required.")
        root = Path(self.package_root).expanduser().resolve(strict=True)
        paths = dict(self.implementation_files)
        if len(paths) != len(self.implementation_files) or not paths:
            raise ValueError("Runtime implementation pins must be nonempty and unique.")
        for path, digest in paths.items():
            if not Path(path).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(
                    "Runtime pins require absolute paths and SHA-256 digests."
                )
        if any(
            str(root / file) not in paths
            for file in ("pyDAFoam.py", "mphys/mphys_dafoam.py")
        ):
            raise ValueError("Pin both pyDAFoam.py and the real MPhys implementation.")
        object.__setattr__(self, "package_root", str(root))


def pin_dafoam_runtime(
    python: PinnedExecutable,
    package_root: str | os.PathLike[str],
    *,
    license_id: str,
    dependency_roots: Sequence[str | os.PathLike[str]] = (),
    max_files: int = 100000,
    max_bytes: int = 8 * 1024**3,
) -> DAFoamRuntime:
    """Read and hash installed implementation bytes without importing an engine.

    Explicit dependency roots extend the pin; they never install or download
    anything. Directory symlinks are rejected rather than silently followed.
    """
    _host_only(max_files, max_bytes)
    if max_files <= 0 or max_bytes <= 0:
        raise ValueError("Runtime identity bounds must be positive.")
    root = Path(package_root).expanduser().resolve(strict=True)
    identities = {}
    total = 0
    for directory in (
        root,
        *(Path(p).expanduser().resolve(strict=True) for p in dependency_roots),
    ):
        if not directory.is_dir():
            raise ValueError("Runtime roots must be directories.")
        for base, dirs, files in os.walk(directory, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".git"))
            if any((Path(base) / d).is_symlink() for d in dirs):
                raise ValueError(
                    "Runtime directory symlinks must be pinned as explicit roots."
                )
            for name in sorted(files):
                if not (
                    name.endswith((".py", ".so", ".dylib", ".dll", ".pyd"))
                    or ".so." in name
                ):
                    continue
                path = Path(base) / name
                if path.is_symlink():
                    # Shared-library soname links may point at the same real file.
                    path = path.resolve(strict=True)
                if str(path) in identities:
                    continue
                total += path.stat().st_size
                if len(identities) >= max_files or total > max_bytes:
                    raise ValueError("Runtime implementation exceeds identity bounds.")
                identities[str(path)] = _digest_file(path)
    if not any(
        str(root) + os.sep in p and (".so" in p or p.endswith(".dylib"))
        for p in identities
    ):
        raise ValueError("DAFoam has no built native extension under package_root.")
    return DAFoamRuntime(python, str(root), tuple(sorted(identities.items())), license_id)


@dataclass(frozen=True, slots=True)
class DAFoamAdjointEvidence:
    function: str
    accepted: bool
    state_sha256: str
    mesh_sha256: str
    design_sha256: str
    adjoint_sha256: str
    linear_iterations: int
    linear_residual_norm: float | None
    petsc_converged_reason: int
    failure_reason: str


@dataclass(frozen=True, slots=True)
class DAFoamTotalDerivative:
    function: str
    design_variable: str
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DAFoamResult:
    """External functionals and optional accepted-point *total* adjoints.

    Mesh/state acceptance is engine-native checkMesh/primalFail evidence under
    declared options, not a fabricated native residual tolerance. Failed primal
    solves expose no objective values; failed adjoints expose no total gradients.
    """

    functions: tuple[tuple[str, float], ...]
    total_derivatives: tuple[DAFoamTotalDerivative, ...]
    mesh_accepted: bool
    state_accepted: bool
    state_sha256: str
    mesh_sha256: str
    design_sha256: str
    adjoints: tuple[DAFoamAdjointEvidence, ...]
    gradient_capability: str
    failure_reason: str
    run: EnergyRunResult
    artifact: ScientificArtifactEnvelope

    @property
    def accepted(self) -> bool:
        return (
            self.mesh_accepted
            and self.state_accepted
            and not self.failure_reason
            and all(a.accepted for a in self.adjoints)
        )

    def require_acceptance(self) -> DAFoamResult:
        if not self.accepted:
            raise DAFoamConvergenceError(self)
        return self


class DAFoamConvergenceError(RuntimeError):
    def __init__(self, result: DAFoamResult):
        self.result = result
        super().__init__(result.failure_reason or "DAFoam acceptance failed.")


def _request(
    case_files: Mapping[str, bytes],
    options: Mapping,
    design: Mapping[str, Sequence[float]],
    geometry_source: str,
    total_adjoint: bool,
    max_bytes: int,
) -> tuple[dict[str, bytes], dict]:
    if not geometry_source.strip():
        raise ValueError("Declare the case geometry source/provenance.")
    if type(total_adjoint) is not bool:
        raise TypeError("total_adjoint must be a bool.")
    files = {}
    total = 0
    for name, data in case_files.items():
        name = _relative_path(name)
        if name.split("/")[0] not in ("0", "constant", "system"):
            raise ValueError(
                "Only initial fields, constant data and system dictionaries are case inputs."
            )
        if not isinstance(data, bytes):
            raise TypeError("Case files must contain exact bytes.")
        total += len(data)
        if total > max_bytes or len(files) >= 100000:
            raise ValueError("Case inputs exceed resource bounds.")
        files[name] = data
    required = {
        "constant/polyMesh/" + p
        for p in ("points", "faces", "owner", "neighbour", "boundary")
    }
    required.update("system/" + p for p in ("controlDict", "fvSchemes", "fvSolution"))
    if not required.issubset(files):
        raise ValueError(
            f"Missing prepared OpenFOAM case files: {sorted(required - files.keys())}"
        )
    opts = json.loads(canonical_json(dict(options)))
    if opts.get("solverName") not in (
        "DASimpleFoam",
        "DARhoSimpleFoam",
        "DARhoSimpleCFoam",
    ):
        raise ValueError("This profile supports steady aerodynamic DAFoam solvers only.")
    if opts.get("useAD", {"mode": "reverse"}).get("mode") != "reverse":
        raise ValueError("The external total-adjoint profile requires reverse AD.")
    if opts.get("useMeanStates", False):
        raise ValueError(
            "Mean-state derivatives are not same-realization steady adjoints."
        )
    if opts.get("adjEqnSolMethod", "Krylov") != "Krylov":
        raise ValueError(
            "This profile requires Krylov adjoints with PETSc convergence evidence."
        )
    if opts.get("discipline", "aero") != "aero":
        raise ValueError("The aerodynamic profile requires discipline='aero'.")
    for key in ("primalMinResTol", "primalMinResTolDiff"):
        if key not in opts or not math.isfinite(opts[key]) or opts[key] <= 0:
            raise ValueError(
                f"Declare a positive finite {key}; do not silently accept engine defaults."
            )
    if (
        not opts.get("checkMeshThreshold")
        or not opts.get("function")
        or not opts.get("designSurfaces")
    ):
        raise ValueError("Declare functions, designSurfaces and mesh-quality thresholds.")
    input_info = opts.get("inputInfo", {})
    if set(input_info) != set(design) or not design:
        raise ValueError("Supply exactly every declared solver input as a design vector.")
    values = {}
    names = tuple(opts["function"])
    for name, vector in design.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in (
            *names,
            "aero_states",
        ):
            raise ValueError(
                "Design-variable names must be distinct OpenMDAO identifiers."
            )
        entry = input_info[name]
        if set(entry.get("components", ())) != {"solver", "function"}:
            raise ValueError(
                "Each input must participate in solver and function derivatives."
            )
        if entry.get("type") not in ("volCoord", "patchVelocity", "patchVar", "field"):
            raise ValueError(
                "Supported design inputs: volCoord, patchVelocity, patchVar, field."
            )
        values[name] = [float(v) for v in vector]
        if not values[name] or not all(math.isfinite(v) for v in values[name]):
            raise ValueError("Design vectors must be nonempty and finite.")
    if any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", n) or n == "aero_states" for n in names
    ):
        raise ValueError("Functions require valid, nonreserved OpenMDAO identifiers.")
    request = {
        "options": opts,
        "design": values,
        "geometry_source": geometry_source,
        "total_adjoint": total_adjoint,
        "case_sha256": {
            p: hashlib.sha256(v).hexdigest() for p, v in sorted(files.items())
        },
    }
    request["request_id"] = canonical_fingerprint(request)
    return files, request


def run_dafoam(
    runtime: DAFoamRuntime,
    *,
    case_files: Mapping[str, bytes],
    options: Mapping,
    design: Mapping[str, Sequence[float]],
    geometry_source: str,
    total_adjoint: bool = False,
    timeout: float = 1800,
    max_output_bytes: int = 128 * 1024 * 1024,
    environment: Mapping[str, str] | None = None,
) -> DAFoamResult:
    """Run a declared, prepared serial OpenFOAM case and optional total adjoints.

    A volCoord design is the complete ordered volume-coordinate vector; its
    totals are with respect to those coordinates, not a pretend CAD/FFD map.
    Patch inputs use DAFoam's declared units/order (patchVelocity: speed, degrees).
    OpenFOAM dictionaries can load native code and are trusted executable inputs.
    All model files are staged privately; no caller script or callback is run.
    """
    _host_only(case_files, options, design, timeout, total_adjoint)
    files, request = _request(
        case_files, options, design, geometry_source, total_adjoint, max_output_bytes
    )
    worker = Path(__file__).with_name("_dafoam_worker.py").read_bytes()
    request["runtime"] = {
        "package_root": runtime.package_root,
        "version": runtime.version,
        "implementation_files": runtime.implementation_files,
    }
    files["aerodynamic-request.json"] = canonical_json(request).encode()
    files["aerodynamic-worker.py"] = worker
    run = run_energy_command(
        runtime.python,
        ("-B", "aerodynamic-worker.py"),
        inputs=files,
        outputs=("aerodynamic-result.json",),
        timeout=timeout,
        max_output_bytes=max_output_bytes,
        environment={**dict(environment or {}), "OPENMDAO_REPORTS": "0", "LC_ALL": "C"},
    )
    payload = json.loads(run.output("aerodynamic-result.json"))
    if payload["request_id"] != request["request_id"]:
        raise ValueError("DAFoam result belongs to a different case/design request.")
    if payload["runtime_sha256"] != canonical_fingerprint(
        dict(runtime.implementation_files)
    ):
        raise ValueError("DAFoam runtime identity does not match its pin.")
    functions = tuple(
        (name, float(value)) for name, value in payload["functions"].items()
    )
    derivatives = tuple(
        DAFoamTotalDerivative(t["function"], t["design_variable"], tuple(t["values"]))
        for t in payload["total_derivatives"]
    )
    adjoints = tuple(DAFoamAdjointEvidence(**a) for a in payload["adjoints"])
    if payload["design_sha256"] != canonical_fingerprint(request["design"]):
        raise ValueError("DAFoam evaluated a different declared design.")
    if not payload["failure_reason"]:
        if not payload["state_accepted"] or not payload["mesh_accepted"]:
            raise ValueError("DAFoam omitted a failed-state/mesh diagnostic.")
        if set(payload["functions"]) != set(request["options"]["function"]):
            raise ValueError("DAFoam omitted requested physical function values.")
        if total_adjoint:
            expected = {
                (name, variable) for name in payload["functions"] for variable in design
            }
            actual = {(t.function, t.design_variable) for t in derivatives}
            if actual != expected or len(actual) != len(derivatives):
                raise ValueError(
                    "DAFoam did not produce every requested total derivative."
                )
            if {a.function for a in adjoints} != set(payload["functions"]) or len(
                adjoints
            ) != len(payload["functions"]):
                raise ValueError(
                    "DAFoam omitted per-functional adjoint convergence evidence."
                )
            if any(
                len(t.values) != len(request["design"][t.design_variable])
                for t in derivatives
            ):
                raise ValueError(
                    "DAFoam total derivative has the wrong design dimension."
                )
    if not total_adjoint and (derivatives or adjoints):
        raise ValueError(
            "Unrequested adjoint evidence cannot be attached to a primal-only run."
        )
    if any(not math.isfinite(v) for _, v in functions) or any(
        not math.isfinite(v) for t in derivatives for v in t.values
    ):
        raise ValueError("DAFoam returned nonfinite physics/derivatives.")
    if functions and (not payload["state_accepted"] or not payload["mesh_accepted"]):
        raise ValueError("Failed primal/mesh cannot expose accepted function values.")
    if derivatives and (
        not all(a.accepted for a in adjoints) or len(adjoints) != len(options["function"])
    ):
        raise ValueError(
            "Total derivatives require accepted evidence for every functional."
        )
    for a in adjoints:
        if (a.state_sha256, a.mesh_sha256, a.design_sha256) != (
            payload["state_sha256"],
            payload["mesh_sha256"],
            payload["design_sha256"],
        ):
            raise ValueError("Adjoint and primal have different realizations.")
    artifact = _artifact(
        "dafoam-aerodynamic-design",
        payload,
        producer="DAFoam",
        version=runtime.version,
        build_id=payload["runtime_sha256"],
        license_id=runtime.license_id,
        resource_id=request["request_id"],
        error=payload["failure_reason"],
        parents=(run.artifact.artifact_id,),
    )
    return DAFoamResult(
        functions,
        derivatives,
        payload["mesh_accepted"],
        payload["state_accepted"],
        payload["state_sha256"],
        payload["mesh_sha256"],
        payload["design_sha256"],
        adjoints,
        "external-total-adjoint" if total_adjoint else "none",
        payload["failure_reason"],
        run,
        artifact,
    )


__all__ = [
    "DAFoamAdjointEvidence",
    "DAFoamConvergenceError",
    "DAFoamResult",
    "DAFoamRuntime",
    "DAFoamTotalDerivative",
    "pin_dafoam_runtime",
    "run_dafoam",
]
