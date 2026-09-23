#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-only pinned PMP preprocessing, SDPB execution, parsing, and audit."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

import numpy as np

from ..._external_runtime import (
    EnergyRunResult,
    EnergyRuntimeError,
    PinnedExecutable,
    run_energy_command,
)
from ..._fingerprint import canonical_fingerprint
from ._pmp import (
    audit_pmp_samples,
    ConformalPolynomialMatrixProgram,
    PMPSampledAuditEvidence,
)


_DECIMAL_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_OUTPUT_PATTERNS = {
    "terminate_reason": re.compile(
        r'^\s*terminateReason\s*=\s*"([^"]+)"\s*;\s*$', re.MULTILINE
    ),
    "primal_objective": re.compile(
        rf"^\s*primalObjective\s*=\s*({_DECIMAL_PATTERN})\s*;\s*$", re.MULTILINE
    ),
    "dual_objective": re.compile(
        rf"^\s*dualObjective\s*=\s*({_DECIMAL_PATTERN})\s*;\s*$", re.MULTILINE
    ),
    "duality_gap": re.compile(
        rf"^\s*dualityGap\s*=\s*({_DECIMAL_PATTERN})\s*;\s*$", re.MULTILINE
    ),
    "primal_error": re.compile(
        rf"^\s*primalError\s*=\s*({_DECIMAL_PATTERN})\s*;\s*$", re.MULTILINE
    ),
    "dual_error": re.compile(
        rf"^\s*dualError\s*=\s*({_DECIMAL_PATTERN})\s*;\s*$", re.MULTILINE
    ),
}


@dataclass(frozen=True, slots=True)
class SDPBProvider:
    """Exact preprocessing and solver executables from one declared release."""

    pmp2sdp: PinnedExecutable
    sdpb: PinnedExecutable

    def __post_init__(self) -> None:
        if not isinstance(self.pmp2sdp, PinnedExecutable) or not isinstance(
            self.sdpb, PinnedExecutable
        ):
            raise TypeError("SDPB provider executables must be PinnedExecutable values.")
        if (
            self.pmp2sdp.version != self.sdpb.version
            or self.pmp2sdp.license_id != self.sdpb.license_id
        ):
            raise ValueError("PMP preprocessing and SDPB releases/licenses must match.")


@dataclass(frozen=True, slots=True)
class SDPBJobPlan:
    """Bounded external solve request with exact decimal thresholds."""

    provider: SDPBProvider
    precision_bits: int
    maximum_iterations: int
    timeout_seconds: float
    maximum_output_bytes: int
    duality_gap_threshold: str
    primal_error_threshold: str
    dual_error_threshold: str
    find_primal_feasible: bool = False
    find_dual_feasible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.provider, SDPBProvider):
            raise TypeError("provider must be SDPBProvider.")
        for name, value in (
            ("precision_bits", self.precision_bits),
            ("maximum_iterations", self.maximum_iterations),
            ("maximum_output_bytes", self.maximum_output_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        for name, value in (
            ("find_primal_feasible", self.find_primal_feasible),
            ("find_dual_feasible", self.find_dual_feasible),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be a bool.")
        if isinstance(self.timeout_seconds, bool):
            raise TypeError("timeout_seconds must be a real duration.")
        if self.precision_bits < 64 or self.maximum_iterations < 1:
            raise ValueError("SDPB precision and iteration limits are invalid.")
        if (
            not np.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0.0
            or self.maximum_output_bytes < 1
        ):
            raise ValueError("SDPB host resource limits are invalid.")
        for name in (
            "duality_gap_threshold",
            "primal_error_threshold",
            "dual_error_threshold",
        ):
            value = getattr(self, name)
            try:
                parsed = Decimal(value)
            except InvalidOperation as error:
                raise ValueError(f"{name} must be an exact decimal string.") from error
            if not parsed.is_finite() or parsed <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.find_primal_feasible and self.find_dual_feasible:
            raise ValueError("Select at most one early feasibility termination mode.")


@dataclass(frozen=True, slots=True)
class SDPBNumericalSummary:
    terminate_reason: str
    primal_objective: str
    dual_objective: str
    duality_gap: str
    primal_error: str
    dual_error: str

    def decimal(self, field: str, /) -> Decimal:
        if field not in {
            "primal_objective",
            "dual_objective",
            "duality_gap",
            "primal_error",
            "dual_error",
        }:
            raise ValueError("Unknown SDPB numerical summary field.")
        return Decimal(getattr(self, field))


SDPBExecutionStatus = Literal[
    "conversion-failed",
    "solver-failed",
    "invalid-output",
    "optimal-audited",
    "feasible-audited",
    "numerically-inconclusive",
    "audit-failed",
    "audit-unavailable",
]


@dataclass(frozen=True, slots=True)
class SDPBExecutionResult:
    program_id: str
    job_id: str
    status: SDPBExecutionStatus
    converter_run: EnergyRunResult | None
    solver_run: EnergyRunResult | None
    summary: SDPBNumericalSummary | None
    functional: tuple[str, ...]
    sampled_audit: PMPSampledAuditEvidence | None
    numerically_accepted: bool
    independently_audited: bool
    successful: bool
    result_id: str
    claim: str = "external-finite-pmp-result-not-continuum-bootstrap-exclusion"


def parse_sdpb_output(data: bytes, /) -> SDPBNumericalSummary:
    """Parse the documented exact-decimal ``out.txt`` summary."""
    text = data.decode("ascii", errors="strict")
    values: dict[str, str] = {}
    for field, pattern in _OUTPUT_PATTERNS.items():
        matches = pattern.findall(text)
        if len(matches) != 1:
            raise ValueError(f"SDPB output must contain exactly one {field} field.")
        values[field] = matches[0]
    for field in (
        "primal_objective",
        "dual_objective",
        "duality_gap",
        "primal_error",
        "dual_error",
    ):
        parsed = Decimal(values[field])
        if not parsed.is_finite():
            raise ValueError("SDPB numerical summary values must be finite.")
    return SDPBNumericalSummary(**values)


def parse_sdpb_vector(data: bytes, /) -> tuple[str, ...]:
    """Parse an SDPB text matrix and require one row or one column."""
    lines = [
        line.strip()
        for line in data.decode("ascii", errors="strict").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError("An SDPB vector file cannot be empty.")
    header = lines[0].split()
    if len(header) != 2 or any(not value.isdigit() for value in header):
        raise ValueError("An SDPB vector header must contain two dimensions.")
    rows, columns = (int(value) for value in header)
    if rows < 1 or columns < 1 or (rows != 1 and columns != 1):
        raise ValueError("An SDPB functional must be a row or column vector.")
    tokens = [token for line in lines[1:] for token in line.split()]
    if len(tokens) != rows * columns:
        raise ValueError("SDPB vector payload size contradicts its header.")
    result = []
    for token in tokens:
        try:
            value = Decimal(token)
        except InvalidOperation as error:
            raise ValueError("SDPB vector contains an invalid decimal.") from error
        if not value.is_finite():
            raise ValueError("SDPB vector values must be finite.")
        result.append(token)
    return tuple(result)


def reconstruct_pmp_functional(
    program: ConformalPolynomialMatrixProgram,
    free_values: Sequence[str],
    /,
) -> tuple[str, ...]:
    """Undo pmp2sdp's normalization elimination using the first nonzero pivot."""
    if not isinstance(program, ConformalPolynomialMatrixProgram):
        raise TypeError("program must be ConformalPolynomialMatrixProgram.")
    free = tuple(Decimal(str(value)) for value in free_values)
    if len(free) != program.functional_count - 1 or any(
        not value.is_finite() for value in free
    ):
        raise ValueError("SDPB free functional dimension is inconsistent with the PMP.")
    normalization = tuple(Decimal(value) for value in program.normalization)
    pivot = next((index for index, value in enumerate(normalization) if value != 0), None)
    if pivot is None:
        raise ValueError("PMP normalization has no nonzero pivot.")
    functional = [Decimal(0)] * program.functional_count
    free_cursor = 0
    for index in range(program.functional_count):
        if index == pivot:
            continue
        functional[index] = free[free_cursor]
        free_cursor += 1
    remainder = sum(
        normalization[index] * functional[index]
        for index in range(program.functional_count)
        if index != pivot
    )
    functional[pivot] = (Decimal(1) - remainder) / normalization[pivot]
    return tuple(str(value.normalize()) for value in functional)


def _job_id(program: ConformalPolynomialMatrixProgram, job: SDPBJobPlan, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "sdpb-job-plan",
            "program": program.pmp_id,
            "pmp2sdp_sha256": job.provider.pmp2sdp.sha256,
            "sdpb_sha256": job.provider.sdpb.sha256,
            "precision_bits": job.precision_bits,
            "maximum_iterations": job.maximum_iterations,
            "timeout_seconds": job.timeout_seconds,
            "maximum_output_bytes": job.maximum_output_bytes,
            "duality_gap_threshold": job.duality_gap_threshold,
            "primal_error_threshold": job.primal_error_threshold,
            "dual_error_threshold": job.dual_error_threshold,
            "find_primal_feasible": job.find_primal_feasible,
            "find_dual_feasible": job.find_dual_feasible,
        }
    )


def _failed_run(error: EnergyRuntimeError, /) -> EnergyRunResult | None:
    return error.result


def _result(
    program: ConformalPolynomialMatrixProgram,
    job_id: str,
    status: SDPBExecutionStatus,
    converter: EnergyRunResult | None,
    solver: EnergyRunResult | None,
    summary: SDPBNumericalSummary | None,
    functional: tuple[str, ...],
    audit: PMPSampledAuditEvidence | None,
    numerically_accepted: bool,
    independently_audited: bool,
    /,
) -> SDPBExecutionResult:
    successful = numerically_accepted and independently_audited
    identifier = canonical_fingerprint(
        {
            "kind": "sdpb-execution-result",
            "program": program.pmp_id,
            "job": job_id,
            "status": status,
            "converter_artifact": None
            if converter is None
            else converter.artifact.artifact_id,
            "solver_artifact": None if solver is None else solver.artifact.artifact_id,
            "summary": None
            if summary is None
            else {
                "terminate_reason": summary.terminate_reason,
                "primal_objective": summary.primal_objective,
                "dual_objective": summary.dual_objective,
                "duality_gap": summary.duality_gap,
                "primal_error": summary.primal_error,
                "dual_error": summary.dual_error,
            },
            "functional": functional,
            "audit": None if audit is None else bool(audit.accepted),
            "numerically_accepted": numerically_accepted,
            "independently_audited": independently_audited,
        }
    )
    return SDPBExecutionResult(
        program_id=program.pmp_id,
        job_id=job_id,
        status=status,
        converter_run=converter,
        solver_run=solver,
        summary=summary,
        functional=functional,
        sampled_audit=audit,
        numerically_accepted=numerically_accepted,
        independently_audited=independently_audited,
        successful=successful,
        result_id=identifier,
    )


def execute_sdpb(
    program: ConformalPolynomialMatrixProgram,
    job: SDPBJobPlan,
    /,
) -> SDPBExecutionResult:
    """Run pmp2sdp and SDPB, then independently sample the reconstructed functional."""
    if not isinstance(program, ConformalPolynomialMatrixProgram):
        raise TypeError("program must be ConformalPolynomialMatrixProgram.")
    if not isinstance(job, SDPBJobPlan):
        raise TypeError("job must be SDPBJobPlan.")
    identifier = _job_id(program, job)
    converter: EnergyRunResult | None = None
    try:
        converter = run_energy_command(
            job.provider.pmp2sdp,
            (
                f"--precision={job.precision_bits}",
                "--input=pmp.json",
                "--output=sdp.zip",
                "--zip",
            ),
            inputs={"pmp.json": program.to_json_bytes()},
            outputs=("sdp.zip",),
            timeout=job.timeout_seconds,
            max_output_bytes=job.maximum_output_bytes,
        )
    except EnergyRuntimeError as error:
        return _result(
            program,
            identifier,
            "conversion-failed",
            _failed_run(error),
            None,
            None,
            (),
            None,
            False,
            False,
        )

    arguments = [
        f"--precision={job.precision_bits}",
        "--sdpDir=sdp.zip",
        "--outDir=out",
        "--checkpointDir=checkpoint",
        f"--maxIterations={job.maximum_iterations}",
        f"--dualityGapThreshold={job.duality_gap_threshold}",
        f"--primalErrorThreshold={job.primal_error_threshold}",
        f"--dualErrorThreshold={job.dual_error_threshold}",
    ]
    if job.find_primal_feasible:
        arguments.append("--findPrimalFeasible=true")
    if job.find_dual_feasible:
        arguments.append("--findDualFeasible=true")
    solver: EnergyRunResult | None = None
    try:
        solver = run_energy_command(
            job.provider.sdpb,
            tuple(arguments),
            inputs={"sdp.zip": converter.output("sdp.zip")},
            outputs=("out/out.txt", "out/y.txt"),
            timeout=job.timeout_seconds,
            max_output_bytes=job.maximum_output_bytes,
        )
    except EnergyRuntimeError as error:
        return _result(
            program,
            identifier,
            "solver-failed",
            converter,
            _failed_run(error),
            None,
            (),
            None,
            False,
            False,
        )

    try:
        summary = parse_sdpb_output(solver.output("out/out.txt"))
        free = parse_sdpb_vector(solver.output("out/y.txt"))
        functional = reconstruct_pmp_functional(program, free)
    except (UnicodeDecodeError, ValueError, KeyError):
        return _result(
            program,
            identifier,
            "invalid-output",
            converter,
            solver,
            None,
            (),
            None,
            False,
            False,
        )

    gap_ok = summary.decimal("duality_gap") <= Decimal(job.duality_gap_threshold)
    primal_ok = summary.decimal("primal_error") <= Decimal(job.primal_error_threshold)
    dual_ok = summary.decimal("dual_error") <= Decimal(job.dual_error_threshold)
    reason = summary.terminate_reason.lower()
    optimal = "primal-dual optimal" in reason and gap_ok and primal_ok and dual_ok
    feasible = (
        job.find_primal_feasible and "primal feasible" in reason and primal_ok
    ) or (job.find_dual_feasible and "dual feasible" in reason and dual_ok)
    numerically_accepted = bool(optimal or feasible)

    points = sorted(
        {
            float(Decimal(point))
            for block in program.blocks
            for point in block.sample_points
        }
    )
    audit = None
    independently_audited = False
    if points:
        audit = audit_pmp_samples(
            program,
            np.asarray([float(Decimal(value)) for value in functional]),
            np.asarray(points),
            tolerance=max(
                float(Decimal(job.primal_error_threshold)),
                float(Decimal(job.dual_error_threshold)),
            ),
        )
        independently_audited = bool(audit.accepted)
    if not numerically_accepted:
        status: SDPBExecutionStatus = "numerically-inconclusive"
    elif audit is None:
        status = "audit-unavailable"
    elif not independently_audited:
        status = "audit-failed"
    elif optimal:
        status = "optimal-audited"
    else:
        status = "feasible-audited"
    return _result(
        program,
        identifier,
        status,
        converter,
        solver,
        summary,
        functional,
        audit,
        numerically_accepted,
        independently_audited,
    )


__all__ = [
    "SDPBExecutionResult",
    "SDPBExecutionStatus",
    "SDPBJobPlan",
    "SDPBNumericalSummary",
    "SDPBProvider",
    "execute_sdpb",
    "parse_sdpb_output",
    "parse_sdpb_vector",
    "reconstruct_pmp_functional",
]
