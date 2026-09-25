#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host-only DAFoam 4.0.3 steady MPhys profile with same-realization totals.

This adapter launches our bundled worker using a pinned Python executable. The
worker invokes PYDAFOAM, DAFoamSolver, DAFoamFunctions and OpenMDAO compute_totals;
its JSON is transport produced by PHYDRAX, not an alleged DAFoam file format.
No native PHYDRAX residual or differentiation through a host process is exposed;
`DAFoamAdjointAction` stages the accepted totals as a host-boundary adjoint.
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
from enum import StrEnum
from pathlib import Path
from typing import Any, final

import numpy as np

from .._external_runtime import (
    _artifact,
    _host_only,
    EnergyRunResult,
    ExternalAdjointAction,
    ExternalPrimalStage,
    ExternalTensorSpec,
    PinnedExecutable,
    run_energy_command,
)
from .._external_worker import _digest_file, _relative_path
from .._fingerprint import canonical_fingerprint, canonical_json
from ..artifacts import ScientificArtifactEnvelope


class DAFoamDesignVariableKind(StrEnum):
    """Closed set of DAFoam solver-input kinds of the steady aerodynamic profile.

    Values are DAFoam's own `inputInfo` type spellings.
    """

    VOLUME_COORDINATES = "volCoord"
    PATCH_VELOCITY = "patchVelocity"
    PATCH_VARIABLE = "patchVar"
    FIELD = "field"


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


@dataclass(frozen=True, slots=True, eq=False)
class DAFoamTotalDerivative:
    """Total derivative of one function with respect to one design variable.

    `values` is read-only and has the declared shape of the design variable.
    """

    function: str
    design_variable: str
    values: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class DAFoamResult:
    """External functionals and optional accepted-point *total* adjoints.

    Mesh/state acceptance is engine-native checkMesh/primalFail evidence under
    declared options, not a fabricated native residual tolerance. Failed primal
    solves expose no objective values; failed adjoints expose no total gradients.
    Functions and totals are in canonical (function, design variable) name
    order. `replay_id` identifies the realization: request, runtime, state,
    mesh, design, and adjoint digests.
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
    replay_id: str
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


def _design_values(
    design: Mapping[str, Any], /
) -> tuple[dict[str, list[float]], dict[str, list[int]]]:
    values = {}
    shapes = {}
    for name in sorted(design):
        array = np.asarray(design[name], dtype=np.float64)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError("Design vectors must be nonempty and finite.")
        values[name] = array.reshape(-1).tolist()
        shapes[name] = list(array.shape)
    return values, shapes


def _request(
    case_files: Mapping[str, bytes],
    options: Mapping,
    design: Mapping[str, Any],
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
    names = tuple(sorted(opts["function"]))
    for name in design:
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
        if entry.get("type") not in tuple(DAFoamDesignVariableKind):
            raise ValueError(
                "Supported design inputs: "
                f"{', '.join(kind.value for kind in DAFoamDesignVariableKind)}."
            )
    values, shapes = _design_values(design)
    if any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", n) or n == "aero_states" for n in names
    ):
        raise ValueError("Functions require valid, nonreserved OpenMDAO identifiers.")
    request = {
        "options": opts,
        "design": values,
        "design_shapes": shapes,
        "geometry_source": geometry_source,
        "total_adjoint": total_adjoint,
        "case_sha256": {
            p: hashlib.sha256(v).hexdigest() for p, v in sorted(files.items())
        },
    }
    request["request_id"] = canonical_fingerprint(request)
    return files, request


def _total_derivatives(
    payload: Mapping[str, Any], request: Mapping[str, Any], /
) -> tuple[DAFoamTotalDerivative, ...]:
    """Shape-preserving totals in canonical (function, design variable) order."""
    derivatives = []
    for total in payload["total_derivatives"]:
        variable = total["design_variable"]
        if variable not in request["design_shapes"]:
            raise ValueError("DAFoam returned a total for an undeclared design input.")
        values = np.asarray(total["values"], dtype=np.float64)
        if values.shape != (len(request["design"][variable]),):
            raise ValueError("DAFoam total derivative has the wrong design dimension.")
        if not np.all(np.isfinite(values)):
            raise ValueError("DAFoam returned nonfinite physics/derivatives.")
        values = values.reshape(request["design_shapes"][variable])
        values.setflags(write=False)
        derivatives.append(DAFoamTotalDerivative(total["function"], variable, values))
    return tuple(
        sorted(derivatives, key=lambda total: (total.function, total.design_variable))
    )


def _replay_id(
    request: Mapping[str, Any],
    payload: Mapping[str, Any],
    adjoints: tuple[DAFoamAdjointEvidence, ...],
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "dafoam-realization",
            "request_id": request["request_id"],
            "runtime_sha256": payload["runtime_sha256"],
            "state_sha256": payload["state_sha256"],
            "mesh_sha256": payload["mesh_sha256"],
            "design_sha256": payload["design_sha256"],
            "adjoints": sorted([a.function, a.adjoint_sha256] for a in adjoints),
        }
    )


def run_dafoam(
    runtime: DAFoamRuntime,
    *,
    case_files: Mapping[str, bytes],
    options: Mapping,
    design: Mapping[str, Any],
    geometry_source: str,
    total_adjoint: bool = False,
    timeout: float = 1800,
    max_output_bytes: int = 128 * 1024 * 1024,
    environment: Mapping[str, str] | None = None,
) -> DAFoamResult:
    """Run a declared, prepared serial OpenFOAM case and optional total adjoints.

    Each design value is an array whose flattened C-order values form DAFoam's
    input vector; its total derivatives keep that array's shape. A volCoord
    design is the complete ordered volume-coordinate vector; its totals are with
    respect to those coordinates, not a pretend CAD/FFD map. Patch inputs use
    DAFoam's declared units/order (patchVelocity: speed, degrees).
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
        (name, float(value)) for name, value in sorted(payload["functions"].items())
    )
    derivatives = _total_derivatives(payload, request)
    adjoints = tuple(
        sorted(
            (DAFoamAdjointEvidence(**a) for a in payload["adjoints"]),
            key=lambda adjoint: adjoint.function,
        )
    )
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
    if not total_adjoint and (derivatives or adjoints):
        raise ValueError(
            "Unrequested adjoint evidence cannot be attached to a primal-only run."
        )
    if any(not math.isfinite(v) for _, v in functions):
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
        _replay_id(request, payload, adjoints),
        run,
        artifact,
    )


def _dafoam_output_schema(
    options: Mapping[str, Any], /
) -> tuple[ExternalTensorSpec, ...]:
    functions = options.get("function")
    if not functions:
        raise ValueError("Declare functions, designSurfaces and mesh-quality thresholds.")
    return tuple(ExternalTensorSpec(name, (), np.float64) for name in sorted(functions))


@final
class DAFoamAdjointAction(ExternalAdjointAction):
    """Staged same-realization DAFoam total adjoint of one prepared case.

    The staged sequence has six steps:

    1. Declare every design variable with its shape. The input schema orders
       them by name and the output schema orders the scalar functions by name.
    2. `stage_primal(*design)` runs the primal and every per-functional total
       adjoint in one pinned run (`run_dafoam(..., total_adjoint=True)`).
    3. Mesh, state, and every adjoint must be accepted; otherwise the stage
       records the failure and evidence and exposes no outputs.
    4. An accepted stage holds the functions, the shape-preserving totals, and
       the run's `replay_id` as its realization.
    5. `apply_adjoint(stage, *ȳ)` refuses a stage of another action or
       realization and a failed stage.
    6. It returns `Jᵀȳ = Σ_f ȳ[f]·totals[f, var]` for every design variable, in
       the variable's declared shape.

    The action is host-only; it never differentiates through the process.
    """

    runtime: DAFoamRuntime
    case_files: Mapping[str, bytes]
    options: Mapping[str, Any]
    geometry_source: str
    timeout: float
    max_output_bytes: int
    environment: Mapping[str, str] | None

    def __init__(
        self,
        runtime: DAFoamRuntime,
        *,
        case_files: Mapping[str, bytes],
        options: Mapping[str, Any],
        design_shapes: Mapping[str, Sequence[int]],
        geometry_source: str,
        timeout: float = 1800,
        max_output_bytes: int = 128 * 1024 * 1024,
        environment: Mapping[str, str] | None = None,
    ):
        _host_only(case_files, options, design_shapes, timeout)
        if not isinstance(runtime, DAFoamRuntime):
            raise TypeError("runtime must be a DAFoamRuntime.")
        # Validate the case and options once against a placeholder design.
        _, request = _request(
            case_files,
            options,
            {name: np.zeros(tuple(shape)) for name, shape in design_shapes.items()},
            geometry_source,
            True,
            max_output_bytes,
        )
        super().__init__(
            provider="DAFoam",
            version=runtime.version,
            input_schema=tuple(
                ExternalTensorSpec(name, tuple(shape), np.float64)
                for name, shape in request["design_shapes"].items()
            ),
            output_schema=_dafoam_output_schema(request["options"]),
            configuration_id=canonical_fingerprint(
                {
                    "runtime": dict(runtime.implementation_files),
                    "case_sha256": request["case_sha256"],
                    "options": request["options"],
                    "geometry_source": geometry_source,
                }
            ),
        )
        self.runtime = runtime
        self.case_files = dict(case_files)
        self.options = request["options"]
        self.geometry_source = geometry_source
        self.timeout = timeout
        self.max_output_bytes = max_output_bytes
        self.environment = None if environment is None else dict(environment)

    def _primal(self, inputs: tuple[np.ndarray, ...], /) -> ExternalPrimalStage:
        result = run_dafoam(
            self.runtime,
            case_files=self.case_files,
            options=self.options,
            design={
                spec.name: value
                for spec, value in zip(self.input_schema, inputs, strict=True)
            },
            geometry_source=self.geometry_source,
            total_adjoint=True,
            timeout=self.timeout,
            max_output_bytes=self.max_output_bytes,
            environment=self.environment,
        )
        evidence = (
            ("artifact", result.artifact.artifact_id),
            ("design_sha256", result.design_sha256),
            ("mesh_sha256", result.mesh_sha256),
            ("run", result.run.artifact.artifact_id),
            ("state_sha256", result.state_sha256),
            *((f"adjoint:{a.function}", a.adjoint_sha256) for a in result.adjoints),
        )
        if not result.accepted:
            return ExternalPrimalStage(
                self.action_id,
                inputs,
                (),
                result.replay_id,
                evidence=evidence,
                failure_reason=result.failure_reason or "DAFoam acceptance failed.",
            )
        totals = {
            (total.function, total.design_variable): total.values
            for total in result.total_derivatives
        }
        return ExternalPrimalStage(
            self.action_id,
            inputs,
            tuple(np.asarray(value, dtype=np.float64) for _, value in result.functions),
            result.replay_id,
            replay_data=tuple(
                (
                    f"totals:{spec.name}",
                    np.stack(
                        [
                            totals[function.name, spec.name]
                            for function in self.output_schema
                        ]
                    ),
                )
                for spec in self.input_schema
            ),
            evidence=evidence,
        )

    def _adjoint(
        self,
        stage: ExternalPrimalStage,
        output_cotangents: tuple[np.ndarray, ...],
        /,
    ) -> tuple[tuple[np.ndarray, ...], str]:
        weights = np.stack(output_cotangents)
        cotangents = []
        for spec in self.input_schema:
            totals = stage.replay(f"totals:{spec.name}")
            cotangents.append(
                (weights @ totals.reshape(weights.size, -1)).reshape(spec.shape)
            )
        return tuple(cotangents), stage.realization_id


__all__ = [
    "DAFoamAdjointAction",
    "DAFoamAdjointEvidence",
    "DAFoamConvergenceError",
    "DAFoamDesignVariableKind",
    "DAFoamResult",
    "DAFoamRuntime",
    "DAFoamTotalDerivative",
    "pin_dafoam_runtime",
    "run_dafoam",
]
