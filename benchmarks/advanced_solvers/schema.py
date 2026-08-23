#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "advanced-solvers/v3"
ROW_STATUSES = frozenset({"success", "nonconverged", "skipped"})
TIMING_PHASES = (
    "setup",
    "compilation",
    "preparation",
    "solve",
    "differentiation_compilation",
    "differentiation",
    "verification",
    "refresh",
    "refreshed_solve",
    "refreshed_verification",
)


class SchemaError(ValueError):
    """A benchmark report is incomplete, internally inconsistent, or invalid."""


def stable_fingerprint(value: Any, /) -> str:
    """Return a stable SHA-256 fingerprint for JSON-compatible evidence."""
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def empty_distribution() -> dict[str, Any]:
    return {
        "count": 0,
        "samples_ms": [],
        "min_ms": None,
        "median_ms": None,
        "mean_ms": None,
        "std_ms": None,
        "max_ms": None,
    }


def skip_certificate(kind: str, /) -> dict[str, Any]:
    return {
        "kind": kind,
        "residual_norm": None,
        "relative_residual": None,
        "backward_error": None,
        "independently_computed": True,
        "evaluator": "benchmarks.advanced_solvers.certificates",
        "details": {},
    }


def validate_report(report: Mapping[str, Any], /) -> None:
    """Validate the stable report schema and all cross-row evidence invariants."""
    _require_keys(
        report,
        ("schema_version", "environment", "campaign", "rows"),
        path="report",
    )
    if report["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(
            f"report.schema_version must be {SCHEMA_VERSION!r}; "
            f"received {report['schema_version']!r}"
        )
    environment = _mapping(report["environment"], "report.environment")
    _validate_environment(environment, "report.environment")
    campaign = _mapping(report["campaign"], "report.campaign")
    _require_keys(
        campaign,
        ("seed", "warmup", "repeats", "selected_adapters", "selected_cases"),
        path="report.campaign",
    )
    _nonnegative_integer(campaign["warmup"], "report.campaign.warmup")
    _positive_integer(campaign["repeats"], "report.campaign.repeats")
    if not isinstance(campaign["seed"], int) or isinstance(campaign["seed"], bool):
        raise SchemaError("report.campaign.seed must be an integer")
    for field in ("selected_adapters", "selected_cases"):
        values = campaign[field]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ):
            raise SchemaError(
                f"report.campaign.{field} must be a non-empty list of names"
            )
        if len(set(values)) != len(values):
            raise SchemaError(f"report.campaign.{field} must not contain duplicates")
    rows = report["rows"]
    if not isinstance(rows, list):
        raise SchemaError("report.rows must be a list")
    if not rows:
        raise SchemaError("report.rows must not be empty")
    expected_protocol = [
        (case_id, adapter)
        for case_id in campaign["selected_cases"]
        for adapter in campaign["selected_adapters"]
    ]
    actual_protocol: list[tuple[str, str]] = []
    identities: set[str] = set()
    for index, row_value in enumerate(rows):
        row = _mapping(row_value, f"report.rows[{index}]")
        validate_row(row, path=f"report.rows[{index}]")
        actual_protocol.append((row["case_id"], row["implementation"]["adapter"]))
        if row["environment"] != environment:
            raise SchemaError(
                f"report.rows[{index}].environment must equal report.environment"
            )
        if (
            row["outcome"]["status"] != "skipped"
            and row["timing"]["solve"]["count"] != campaign["repeats"]
        ):
            raise SchemaError(
                f"report.rows[{index}].timing.solve.count must equal "
                "report.campaign.repeats"
            )
        differentiation_count = row["timing"]["differentiation"]["count"]
        if differentiation_count not in (0, campaign["repeats"]):
            raise SchemaError(
                f"report.rows[{index}].timing.differentiation.count must equal "
                "report.campaign.repeats when differentiation is measured"
            )
        identity = row_identity(row)
        if identity in identities:
            raise SchemaError(f"duplicate benchmark row identity {identity!r}")
        identities.add(identity)

    if actual_protocol != expected_protocol:
        raise SchemaError(
            "report.rows must be the exact selected case×adapter cross-product "
            "in case-major, adapter-minor order"
        )


def validate_row(row: Mapping[str, Any], /, *, path: str = "row") -> None:
    """Validate one row, including skip and independent-certificate invariants."""
    _require_keys(
        row,
        (
            "schema_version",
            "environment",
            "case_id",
            "problem",
            "implementation",
            "sizes",
            "tolerances",
            "outcome",
            "certificate",
            "operations",
            "refresh",
            "memory",
            "transfers",
            "timing",
            "availability",
        ),
        path=path,
    )
    if row["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(f"{path}.schema_version must be {SCHEMA_VERSION!r}")
    _nonempty_string(row["case_id"], f"{path}.case_id")
    _validate_environment(
        _mapping(row["environment"], f"{path}.environment"), f"{path}.environment"
    )
    problem = _mapping(row["problem"], f"{path}.problem")
    _require_keys(
        problem,
        ("family", "name", "variant", "seed", "dtype", "fingerprint", "parameters"),
        path=f"{path}.problem",
    )
    for field in ("family", "name", "variant", "dtype", "fingerprint"):
        _nonempty_string(problem[field], f"{path}.problem.{field}")
    if not isinstance(problem["seed"], int) or isinstance(problem["seed"], bool):
        raise SchemaError(f"{path}.problem.seed must be an integer")
    _mapping(problem["parameters"], f"{path}.problem.parameters")

    implementation = _mapping(row["implementation"], f"{path}.implementation")
    _require_keys(
        implementation,
        ("adapter", "backend", "method", "preconditioner", "versions"),
        path=f"{path}.implementation",
    )
    for field in ("adapter", "backend", "method", "preconditioner"):
        _nonempty_string(implementation[field], f"{path}.implementation.{field}")
    _mapping(implementation["versions"], f"{path}.implementation.versions")

    sizes = _mapping(row["sizes"], f"{path}.sizes")
    _require_keys(
        sizes,
        ("dimension", "rows", "columns", "nnz", "block_size", "right_hand_sides"),
        path=f"{path}.sizes",
    )
    for field in ("dimension", "rows", "columns", "block_size", "right_hand_sides"):
        _positive_integer(sizes[field], f"{path}.sizes.{field}")
    _nonnegative_integer(sizes["nnz"], f"{path}.sizes.nnz")

    tolerances = _mapping(row["tolerances"], f"{path}.tolerances")
    _require_keys(
        tolerances,
        ("relative", "absolute", "max_steps"),
        path=f"{path}.tolerances",
    )
    _nonnegative_finite(tolerances["relative"], f"{path}.tolerances.relative")
    _nonnegative_finite(tolerances["absolute"], f"{path}.tolerances.absolute")
    _positive_integer(tolerances["max_steps"], f"{path}.tolerances.max_steps")

    outcome = _mapping(row["outcome"], f"{path}.outcome")
    _require_keys(
        outcome,
        ("status", "converged", "message", "skip_reason"),
        path=f"{path}.outcome",
    )
    status = outcome["status"]
    if status not in ROW_STATUSES:
        raise SchemaError(f"{path}.outcome.status must be one of {sorted(ROW_STATUSES)}")
    if not isinstance(outcome["message"], str):
        raise SchemaError(f"{path}.outcome.message must be a string")

    availability = _mapping(row["availability"], f"{path}.availability")
    _require_keys(
        availability,
        ("available", "capability", "dependency", "dependency_version", "reason"),
        path=f"{path}.availability",
    )
    if not isinstance(availability["available"], bool):
        raise SchemaError(f"{path}.availability.available must be boolean")
    _nonempty_string(availability["capability"], f"{path}.availability.capability")
    _nonempty_string(availability["dependency"], f"{path}.availability.dependency")

    certificate = _mapping(row["certificate"], f"{path}.certificate")
    _require_keys(
        certificate,
        (
            "kind",
            "residual_norm",
            "relative_residual",
            "backward_error",
            "independently_computed",
            "evaluator",
            "details",
        ),
        path=f"{path}.certificate",
    )
    _nonempty_string(certificate["kind"], f"{path}.certificate.kind")
    certificate_details = _mapping(
        certificate["details"],
        f"{path}.certificate.details",
    )
    if certificate["independently_computed"] is not True:
        raise SchemaError(f"{path}.certificate must be independently computed")
    _nonempty_string(certificate["evaluator"], f"{path}.certificate.evaluator")

    operations = _mapping(row["operations"], f"{path}.operations")
    _require_keys(
        operations,
        (
            "iterations",
            "matvecs",
            "preconditioner_applications",
            "linear_solves",
            "nonlinear_evaluations",
            "jacobian_evaluations",
        ),
        path=f"{path}.operations",
    )
    for field, value in operations.items():
        if value is not None:
            _nonnegative_integer(value, f"{path}.operations.{field}")

    refresh = _mapping(row["refresh"], f"{path}.refresh")
    _require_keys(
        refresh,
        (
            "applicable",
            "symbolic_reused",
            "numeric_refreshed",
            "symbolic_refresh_count",
            "numeric_refresh_count",
            "evidence",
            "certificate_problem_fingerprint",
            "certificate_kind",
            "certificate_relative_residual",
            "certificate_backward_error",
            "certificate_converged",
            "independently_certified",
        ),
        path=f"{path}.refresh",
    )
    if not isinstance(refresh["applicable"], bool):
        raise SchemaError(f"{path}.refresh.applicable must be boolean")
    for field in ("symbolic_refresh_count", "numeric_refresh_count"):
        _nonnegative_integer(refresh[field], f"{path}.refresh.{field}")
    _nonempty_string(refresh["evidence"], f"{path}.refresh.evidence")
    refresh_certificate_fields = (
        "certificate_problem_fingerprint",
        "certificate_kind",
        "certificate_relative_residual",
        "certificate_backward_error",
        "certificate_converged",
        "independently_certified",
    )
    if refresh["applicable"]:
        for field in ("symbolic_reused", "numeric_refreshed"):
            if not isinstance(refresh[field], bool):
                raise SchemaError(
                    f"{path}.refresh.{field} must be boolean when refresh is applicable"
                )
        _nonempty_string(
            refresh["certificate_problem_fingerprint"],
            f"{path}.refresh.certificate_problem_fingerprint",
        )
        _nonempty_string(
            refresh["certificate_kind"],
            f"{path}.refresh.certificate_kind",
        )
        _nonnegative_finite(
            refresh["certificate_relative_residual"],
            f"{path}.refresh.certificate_relative_residual",
        )
        _nonnegative_finite(
            refresh["certificate_backward_error"],
            f"{path}.refresh.certificate_backward_error",
        )
        if not isinstance(refresh["certificate_converged"], bool):
            raise SchemaError(f"{path}.refresh.certificate_converged must be boolean")
        if refresh["independently_certified"] is not True:
            raise SchemaError(f"{path}.refresh.independently_certified must be true")
    elif (
        refresh["symbolic_reused"] is not None
        or refresh["numeric_refreshed"] is not None
        or refresh["symbolic_refresh_count"] != 0
        or refresh["numeric_refresh_count"] != 0
        or any(refresh[field] is not None for field in refresh_certificate_fields)
    ):
        raise SchemaError(
            f"{path}.refresh non-applicable evidence must use null flags and zero counts"
        )

    memory = _mapping(row["memory"], f"{path}.memory")
    _require_keys(
        memory,
        ("matrix_bytes", "setup_bytes", "peak_estimate_bytes", "evidence"),
        path=f"{path}.memory",
    )
    for field in ("matrix_bytes", "setup_bytes", "peak_estimate_bytes"):
        if memory[field] is not None:
            _nonnegative_integer(memory[field], f"{path}.memory.{field}")
    _nonempty_string(memory["evidence"], f"{path}.memory.evidence")

    transfers = _mapping(row["transfers"], f"{path}.transfers")
    _require_keys(
        transfers,
        (
            "input_origin",
            "host_to_device_bytes",
            "host_to_device_timing_phase",
            "device_to_host_bytes",
            "device_to_host_timing_phase",
            "evidence",
        ),
        path=f"{path}.transfers",
    )
    if transfers["input_origin"] != "numpy-host":
        raise SchemaError(
            f"{path}.transfers.input_origin must be the canonical 'numpy-host'"
        )
    _nonempty_string(transfers["evidence"], f"{path}.transfers.evidence")
    for field in ("host_to_device_bytes", "device_to_host_bytes"):
        if transfers[field] is not None:
            _nonnegative_integer(transfers[field], f"{path}.transfers.{field}")

    timing = _mapping(row["timing"], f"{path}.timing")
    _require_keys(timing, TIMING_PHASES, path=f"{path}.timing")
    for phase in TIMING_PHASES:
        _validate_distribution(
            _mapping(timing[phase], f"{path}.timing.{phase}"),
            f"{path}.timing.{phase}",
        )

    if status == "skipped":
        if outcome["converged"] is not None:
            raise SchemaError(f"{path}.outcome.converged must be null for a skip")
        _nonempty_string(outcome["skip_reason"], f"{path}.outcome.skip_reason")
        _nonempty_string(availability["reason"], f"{path}.availability.reason")
        if availability["available"]:
            raise SchemaError(f"{path}.availability.available must be false for a skip")
        for field in ("residual_norm", "relative_residual", "backward_error"):
            if certificate[field] is not None:
                raise SchemaError(f"{path}.certificate.{field} must be null for a skip")
        measured_phases = [
            phase for phase in TIMING_PHASES if timing[phase]["count"] != 0
        ]
        if measured_phases:
            raise SchemaError(
                f"{path}.timing must be empty for a skip; measured "
                f"{', '.join(measured_phases)}"
            )
        if (
            transfers["host_to_device_bytes"] is not None
            or transfers["host_to_device_timing_phase"] is not None
            or transfers["device_to_host_bytes"] is not None
            or transfers["device_to_host_timing_phase"] is not None
        ):
            raise SchemaError(
                f"{path}.transfers must use null counts and phases for a skip"
            )
        if "not measured" not in transfers["evidence"].lower():
            raise SchemaError(
                f"{path}.transfers.evidence must state that skips were not measured"
            )
        return

    if availability["available"] is not True or availability["reason"] is not None:
        raise SchemaError(f"{path}.availability must record an available implementation")
    if outcome["skip_reason"] is not None:
        raise SchemaError(f"{path}.outcome.skip_reason must be null for a measured row")
    transfer_contracts = (
        ("host_to_device_bytes", "host_to_device_timing_phase"),
        ("device_to_host_bytes", "device_to_host_timing_phase"),
    )
    for bytes_field, phase_field in transfer_contracts:
        byte_count = transfers[bytes_field]
        if byte_count is None:
            raise SchemaError(
                f"{path}.transfers.{bytes_field} must be measured for an executed row"
            )
        phase_value = transfers[phase_field]
        if byte_count == 0 and phase_value is not None:
            raise SchemaError(
                f"{path}.transfers.{phase_field} must be null when {bytes_field} is zero"
            )
        if byte_count > 0 and phase_value is None:
            raise SchemaError(
                f"{path}.transfers.{phase_field} must identify measured timing phases "
                f"when {bytes_field} is positive"
            )
        if phase_value is not None:
            phases = phase_value.split("+")
            if any(phase not in TIMING_PHASES for phase in phases) or len(phases) != len(
                set(phases)
            ):
                raise SchemaError(
                    f"{path}.transfers.{phase_field} must contain unique '+'-separated "
                    "timing phase names"
                )
            unmeasured = [phase for phase in phases if timing[phase]["count"] == 0]
            if unmeasured:
                raise SchemaError(
                    f"{path}.transfers.{phase_field} names unmeasured timing phases: "
                    f"{', '.join(unmeasured)}"
                )
    expected_converged = status == "success"
    if outcome["converged"] is not expected_converged:
        raise SchemaError(
            f"{path}.outcome.converged is inconsistent with status {status!r}"
        )
    for field in ("residual_norm", "relative_residual", "backward_error"):
        _nonnegative_finite(certificate[field], f"{path}.certificate.{field}")
    if certificate["kind"] == "continuation-branch-residual":
        continuation_fields = (
            "branch_successful",
            "finite_branch",
            "residuals_satisfied",
            "state_sign_change",
            "tangent_coordinate_sign_change",
            "fold_bracket",
            "successful_fold_traversal",
        )
        _require_keys(
            certificate_details,
            continuation_fields,
            path=f"{path}.certificate.details",
        )
        for field in continuation_fields:
            if not isinstance(certificate_details[field], bool):
                raise SchemaError(f"{path}.certificate.details.{field} must be boolean")
        if status == "success" and not certificate_details["successful_fold_traversal"]:
            raise SchemaError(
                f"{path} cannot report continuation success without independently "
                "certified fold traversal"
            )
    if certificate["kind"] in {"eigenpair-relation", "schur-relation"}:
        eigen_fields = (
            "requested_eigenpairs",
            "returned_eigenpairs",
            "count_satisfied",
            "largest_magnitude_membership_error",
            "membership_tolerance",
            "largest_magnitude_membership_satisfied",
        )
        _require_keys(
            certificate_details,
            eigen_fields,
            path=f"{path}.certificate.details",
        )
        for field in ("requested_eigenpairs", "returned_eigenpairs"):
            _nonnegative_integer(
                certificate_details[field],
                f"{path}.certificate.details.{field}",
            )
        for field in ("count_satisfied", "largest_magnitude_membership_satisfied"):
            if not isinstance(certificate_details[field], bool):
                raise SchemaError(f"{path}.certificate.details.{field} must be boolean")
        if status == "success" and (
            not certificate_details["count_satisfied"]
            or not certificate_details["largest_magnitude_membership_satisfied"]
        ):
            raise SchemaError(
                f"{path} cannot report eigen success without independently "
                "verified requested-count and largest-magnitude membership"
            )
    optimization_fields = {
        "optimization-stationarity": (
            "objective",
            "objective_gap",
            "distance_to_reference",
            "gradient_norm",
        ),
        "optimization-kkt": (
            "objective",
            "objective_gap",
            "distance_to_reference",
            "equality_violation",
            "inequality_violation",
            "estimated_equality_multiplier",
            "dual_stationarity_norm",
        ),
        "optimization-bound-stationarity": (
            "objective",
            "objective_gap",
            "distance_to_reference",
            "projected_stationarity_norm",
            "bound_feasibility",
        ),
        "optimization-proximal-stationarity": (
            "objective",
            "objective_gap",
            "distance_to_reference",
            "proximal_gradient_mapping_norm",
        ),
        "optimization-program-kkt": (
            "objective",
            "objective_gap",
            "distance_to_reference",
            "primal_feasibility",
            "dual_stationarity_norm",
            "cone_violation",
        ),
    }
    if certificate["kind"] in optimization_fields:
        fields = optimization_fields[certificate["kind"]]
        _require_keys(
            certificate_details,
            fields,
            path=f"{path}.certificate.details",
        )
        _finite_number(
            certificate_details["objective"],
            f"{path}.certificate.details.objective",
        )
        if "estimated_equality_multiplier" in fields:
            _finite_number(
                certificate_details["estimated_equality_multiplier"],
                f"{path}.certificate.details.estimated_equality_multiplier",
            )
        for field in fields:
            if field not in {"objective", "estimated_equality_multiplier"}:
                _nonnegative_finite(
                    certificate_details[field],
                    f"{path}.certificate.details.{field}",
                )
    if timing["setup"]["count"] != 1:
        raise SchemaError(f"{path}.timing.setup must contain exactly one sample")
    for phase in (
        "compilation",
        "preparation",
        "differentiation_compilation",
        "refresh",
        "refreshed_solve",
        "refreshed_verification",
    ):
        if timing[phase]["count"] > 1:
            raise SchemaError(f"{path}.timing.{phase} may contain at most one sample")
    expected_differentiation_compilation_count = (
        1 if timing["differentiation"]["count"] > 0 else 0
    )
    if (
        timing["differentiation_compilation"]["count"]
        != expected_differentiation_compilation_count
    ):
        raise SchemaError(
            f"{path}.timing differentiation compilation and execution counts must match"
        )
    expected_refreshed_solve_count = 1 if refresh["applicable"] else 0
    if timing["refreshed_solve"]["count"] != expected_refreshed_solve_count:
        raise SchemaError(
            f"{path}.timing.refreshed_solve count must match refresh applicability"
        )
    if timing["refreshed_verification"]["count"] != expected_refreshed_solve_count:
        raise SchemaError(
            f"{path}.timing.refreshed_verification count must match refresh applicability"
        )
    if timing["solve"]["count"] < 1:
        raise SchemaError(f"{path}.timing.solve must contain measured samples")
    if timing["verification"]["count"] != 1:
        raise SchemaError(f"{path}.timing.verification must contain exactly one sample")


def row_identity(row: Mapping[str, Any], /) -> str:
    """Return the comparison identity for a benchmark row."""
    problem = _mapping(row["problem"], "row.problem")
    implementation = _mapping(row["implementation"], "row.implementation")
    return stable_fingerprint(
        {
            "schema_version": row["schema_version"],
            "case_id": row["case_id"],
            "problem_fingerprint": problem["fingerprint"],
            "adapter": implementation["adapter"],
            "backend": implementation["backend"],
            "method": implementation["method"],
            "preconditioner": implementation["preconditioner"],
            "sizes": row["sizes"],
            "tolerances": row["tolerances"],
        }
    )


def _validate_environment(environment: Mapping[str, Any], path: str) -> None:
    _require_keys(
        environment,
        (
            "fingerprint",
            "python_version",
            "platform",
            "machine",
            "processor",
            "logical_cpus",
            "numpy_version",
            "jax",
            "thread_environment",
        ),
        path=path,
    )
    for field in (
        "fingerprint",
        "python_version",
        "platform",
        "machine",
        "numpy_version",
    ):
        _nonempty_string(environment[field], f"{path}.{field}")
    if not isinstance(environment["processor"], str):
        raise SchemaError(f"{path}.processor must be a string")
    _positive_integer(environment["logical_cpus"], f"{path}.logical_cpus")
    _mapping(environment["jax"], f"{path}.jax")
    _mapping(environment["thread_environment"], f"{path}.thread_environment")


def _validate_distribution(distribution: Mapping[str, Any], path: str) -> None:
    _require_keys(
        distribution,
        ("count", "samples_ms", "min_ms", "median_ms", "mean_ms", "std_ms", "max_ms"),
        path=path,
    )
    _nonnegative_integer(distribution["count"], f"{path}.count")
    samples = distribution["samples_ms"]
    if not isinstance(samples, list):
        raise SchemaError(f"{path}.samples_ms must be a list")
    if len(samples) != distribution["count"]:
        raise SchemaError(f"{path}.count must equal len(samples_ms)")
    for index, sample in enumerate(samples):
        _nonnegative_finite(sample, f"{path}.samples_ms[{index}]")
    summaries = ("min_ms", "median_ms", "mean_ms", "std_ms", "max_ms")
    if not samples:
        if any(distribution[field] is not None for field in summaries):
            raise SchemaError(f"{path} summaries must be null when count is zero")
    else:
        expected = {
            "min_ms": min(samples),
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.fmean(samples),
            "std_ms": statistics.pstdev(samples),
            "max_ms": max(samples),
        }
        for field, expected_value in expected.items():
            _nonnegative_finite(distribution[field], f"{path}.{field}")
            if not math.isclose(
                distribution[field],
                expected_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise SchemaError(
                    f"{path}.{field} does not match samples_ms: "
                    f"expected {expected_value!r}"
                )


def _require_keys(value: Mapping[str, Any], keys: Sequence[str], *, path: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise SchemaError(f"{path} is missing required fields: {', '.join(missing)}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{path} must be an object")
    return value


def _nonempty_string(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{path} must be a non-empty string")


def _finite_number(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{path} must be a finite number")
    if not math.isfinite(float(value)):
        raise SchemaError(f"{path} must be a finite number")


def _nonnegative_finite(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{path} must be a finite non-negative number")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise SchemaError(f"{path} must be a finite non-negative number")


def _positive_integer(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SchemaError(f"{path} must be a positive integer")


def _nonnegative_integer(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{path} must be a non-negative integer")


__all__ = [
    "ROW_STATUSES",
    "SCHEMA_VERSION",
    "TIMING_PHASES",
    "SchemaError",
    "empty_distribution",
    "row_identity",
    "skip_certificate",
    "stable_fingerprint",
    "validate_report",
    "validate_row",
]
