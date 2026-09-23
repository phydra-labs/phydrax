#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from benchmarks._runtime import DurationDistribution
from phydrax._fingerprint import canonical_fingerprint


ROW_STATUSES = frozenset({"success", "nonconverged", "failed", "skipped"})
TIMING_PHASES = (
    "setup",
    "compilation",
    "preparation",
    "warmup",
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


def empty_distribution() -> dict[str, Any]:
    return DurationDistribution(()).to_milliseconds_dict()


def skip_certificate(
    kind: str,
    /,
    *,
    capability: str,
    problem_fingerprint: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "capability": capability,
        "problem_fingerprint": problem_fingerprint,
        "residual_norm": None,
        "relative_residual": None,
        "backward_error": None,
        "independently_computed": False,
        "evaluator": "not executed",
        "evaluator_fingerprint": None,
        "details": {},
    }


def validate_report(report: Mapping[str, Any], /) -> None:
    """Validate the stable report schema and all cross-row evidence invariants."""
    _require_keys(
        report,
        ("environment", "provenance", "campaign", "rows", "passed"),
        path="report",
    )
    environment = _mapping(report["environment"], "report.environment")
    _validate_environment(environment, "report.environment")
    provenance = _mapping(report["provenance"], "report.provenance")
    _require_keys(
        provenance,
        (
            "harness_source_fingerprint",
            "certificate_evaluator_fingerprint",
            "case_source_fingerprint",
        ),
        path="report.provenance",
    )
    for field in provenance:
        _sha256(provenance[field], f"report.provenance.{field}")
    campaign = _mapping(report["campaign"], "report.campaign")
    _require_keys(
        campaign,
        (
            "seed",
            "warmup",
            "repeats",
            "selected_adapters",
            "selected_cases",
            "case_fingerprints",
        ),
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
    case_fingerprints = _mapping(
        campaign["case_fingerprints"], "report.campaign.case_fingerprints"
    )
    if set(case_fingerprints) != set(campaign["selected_cases"]):
        raise SchemaError(
            "report.campaign.case_fingerprints must cover exactly the selected cases"
        )
    for case_id, fingerprint in case_fingerprints.items():
        _sha256(fingerprint, f"report.campaign.case_fingerprints.{case_id}")
    if not isinstance(report["passed"], bool):
        raise SchemaError("report.passed must be boolean")
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
            row["outcome"]["status"] in {"success", "nonconverged"}
            and row["timing"]["solve"]["count"] != campaign["repeats"]
        ):
            raise SchemaError(
                f"report.rows[{index}].timing.solve.count must equal "
                "report.campaign.repeats"
            )
        if (
            row["outcome"]["status"] in {"success", "nonconverged"}
            and row["timing"]["warmup"]["count"] != campaign["warmup"]
        ):
            raise SchemaError(
                f"report.rows[{index}].timing.warmup.count must equal "
                "report.campaign.warmup"
            )
        if row["problem"]["fingerprint"] != case_fingerprints[row["case_id"]]:
            raise SchemaError(
                f"report.rows[{index}].problem fingerprint does not match the "
                "campaign case descriptor"
            )
        if (
            row["outcome"]["status"] in {"success", "nonconverged"}
            and row["certificate"]["evaluator_fingerprint"]
            != provenance["certificate_evaluator_fingerprint"]
        ):
            raise SchemaError(
                f"report.rows[{index}] certificate evaluator source does not match "
                "report provenance"
            )
        differentiation_count = row["timing"]["differentiation"]["count"]
        if (
            row["outcome"]["status"] in {"success", "nonconverged"}
            and differentiation_count not in (0, campaign["repeats"])
        ):
            raise SchemaError(
                f"report.rows[{index}].timing.differentiation.count must equal "
                "report.campaign.repeats when differentiation is measured"
            )
        identity = row_identity(row)
        if identity in identities:
            raise SchemaError(f"duplicate benchmark row identity {identity!r}")
        identities.add(identity)

    expected_passed = all(
        row["outcome"]["status"] in {"success", "skipped"} for row in rows
    )
    if report["passed"] is not expected_passed:
        raise SchemaError("report.passed does not match row outcomes")
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
    _sha256(problem["fingerprint"], f"{path}.problem.fingerprint")
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
        ("status", "converged", "message", "skip_reason", "failure_phase"),
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
            "capability",
            "problem_fingerprint",
            "residual_norm",
            "relative_residual",
            "backward_error",
            "independently_computed",
            "evaluator",
            "evaluator_fingerprint",
            "details",
        ),
        path=f"{path}.certificate",
    )
    _nonempty_string(certificate["kind"], f"{path}.certificate.kind")
    certificate_details = _mapping(
        certificate["details"],
        f"{path}.certificate.details",
    )
    if certificate["capability"] != availability["capability"]:
        raise SchemaError(f"{path}.certificate capability does not match availability")
    if certificate["problem_fingerprint"] != problem["fingerprint"]:
        raise SchemaError(f"{path}.certificate is bound to a different problem")
    if certificate["kind"] not in _certificate_kinds(certificate["capability"]):
        raise SchemaError(
            f"{path}.certificate.kind is invalid for capability "
            f"{certificate['capability']!r}"
        )

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
            if refresh[field] is not None and not isinstance(refresh[field], bool):
                raise SchemaError(
                    f"{path}.refresh.{field} must be boolean or null"
                )
        if refresh["symbolic_reused"] is True and refresh["symbolic_refresh_count"] != 0:
            raise SchemaError(
                f"{path}.refresh symbolic reuse conflicts with symbolic refresh count"
            )
        if refresh["symbolic_reused"] is False and refresh["symbolic_refresh_count"] == 0:
            raise SchemaError(
                f"{path}.refresh symbolic rebuild requires a positive refresh count"
            )
        if refresh["numeric_refreshed"] is not True:
            raise SchemaError(f"{path}.refresh must prove the numeric refresh")
        if refresh["numeric_refresh_count"] != 1:
            raise SchemaError(
                f"{path}.refresh numeric refresh count must equal the measured call"
            )
        _nonempty_string(
            refresh["certificate_problem_fingerprint"],
            f"{path}.refresh.certificate_problem_fingerprint",
        )
        if refresh["certificate_problem_fingerprint"] == problem["fingerprint"]:
            raise SchemaError(f"{path}.refresh must bind a distinct numeric problem")
        _nonempty_string(
            refresh["certificate_kind"],
            f"{path}.refresh.certificate_kind",
        )
        if refresh["certificate_kind"] not in _certificate_kinds(
            certificate["capability"]
        ):
            raise SchemaError(f"{path}.refresh certificate relation is incompatible")
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
        if status == "success" and refresh["certificate_converged"] is not True:
            raise SchemaError(
                f"{path} cannot report lifecycle success when refreshed solve failed"
            )
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
        ("initial", "refreshed", "evidence"),
        path=f"{path}.memory",
    )
    _nonempty_string(memory["evidence"], f"{path}.memory.evidence")
    if memory["initial"] is not None:
        _validate_memory_measurement(
            _mapping(memory["initial"], f"{path}.memory.initial"),
            f"{path}.memory.initial",
        )
    if memory["refreshed"] is not None:
        _validate_memory_measurement(
            _mapping(memory["refreshed"], f"{path}.memory.refreshed"),
            f"{path}.memory.refreshed",
        )

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
        if outcome["failure_phase"] is not None:
            raise SchemaError(f"{path}.outcome.failure_phase must be null for a skip")
        if certificate["independently_computed"] is not False:
            raise SchemaError(f"{path}.certificate cannot be computed for a skip")
        if certificate["evaluator_fingerprint"] is not None:
            raise SchemaError(f"{path}.certificate evaluator must be null for a skip")
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

    if status == "failed":
        if outcome["converged"] is not False:
            raise SchemaError(f"{path}.outcome.converged must be false for a failure")
        if outcome["skip_reason"] is not None:
            raise SchemaError(f"{path}.outcome.skip_reason must be null for a failure")
        _nonempty_string(outcome["failure_phase"], f"{path}.outcome.failure_phase")
        if certificate["independently_computed"] is not False:
            raise SchemaError(f"{path}.certificate cannot be computed for a failure")
        if any(
            certificate[field] is not None
            for field in (
                "residual_norm",
                "relative_residual",
                "backward_error",
                "evaluator_fingerprint",
            )
        ):
            raise SchemaError(f"{path}.certificate must be empty for a failure")
        if (
            transfers["host_to_device_bytes"] is not None
            or transfers["host_to_device_timing_phase"] is not None
            or transfers["device_to_host_bytes"] is not None
            or transfers["device_to_host_timing_phase"] is not None
        ):
            raise SchemaError(f"{path}.transfers must be unknown for a failure")
        return

    if certificate["independently_computed"] is not True:
        raise SchemaError(f"{path}.certificate must be independently computed")
    _nonempty_string(certificate["evaluator"], f"{path}.certificate.evaluator")
    _sha256(
        certificate["evaluator_fingerprint"],
        f"{path}.certificate.evaluator_fingerprint",
    )
    if outcome["failure_phase"] is not None:
        raise SchemaError(f"{path}.outcome.failure_phase must be null when measured")

    if availability["available"] is not True or availability["reason"] is not None:
        raise SchemaError(f"{path}.availability must record an available implementation")
    if outcome["skip_reason"] is not None:
        raise SchemaError(f"{path}.outcome.skip_reason must be null for a measured row")
    if memory["initial"] is None:
        raise SchemaError(f"{path}.memory.initial must be measured for an executed row")
    if refresh["applicable"] != (memory["refreshed"] is not None):
        raise SchemaError(
            f"{path}.memory.refreshed presence must match refresh applicability"
        )
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
            "dual_feasibility",
            "complementarity",
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
    return canonical_fingerprint(
        {
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
            "phydrax_version",
            "platform",
            "machine",
            "processor",
            "logical_cpus",
            "numpy_version",
            "jaxlib_version",
            "default_float_dtype",
            "package_fingerprint",
            "jax",
            "performance_environment",
        ),
        path=path,
    )
    for field in (
        "fingerprint",
        "python_version",
        "phydrax_version",
        "platform",
        "machine",
        "numpy_version",
        "jaxlib_version",
        "default_float_dtype",
        "package_fingerprint",
    ):
        _nonempty_string(environment[field], f"{path}.{field}")
    if not isinstance(environment["processor"], str):
        raise SchemaError(f"{path}.processor must be a string")
    _positive_integer(environment["logical_cpus"], f"{path}.logical_cpus")
    jax_evidence = _mapping(environment["jax"], f"{path}.jax")
    _require_keys(
        jax_evidence,
        ("version", "backend", "x64_enabled", "devices"),
        path=f"{path}.jax",
    )
    _nonempty_string(jax_evidence["version"], f"{path}.jax.version")
    _nonempty_string(jax_evidence["backend"], f"{path}.jax.backend")
    if not isinstance(jax_evidence["x64_enabled"], bool):
        raise SchemaError(f"{path}.jax.x64_enabled must be a boolean")
    devices = jax_evidence["devices"]
    if not isinstance(devices, list) or not devices:
        raise SchemaError(f"{path}.jax.devices must be a non-empty list")
    for index, raw_device in enumerate(devices):
        device_path = f"{path}.jax.devices[{index}]"
        device = _mapping(raw_device, device_path)
        _require_keys(device, ("platform", "kind"), path=device_path)
        _nonempty_string(device["platform"], f"{device_path}.platform")
        _nonempty_string(device["kind"], f"{device_path}.kind")
    performance_environment = _mapping(
        environment["performance_environment"],
        f"{path}.performance_environment",
    )
    for key, value in performance_environment.items():
        _nonempty_string(key, f"{path}.performance_environment key")
        if value is not None and not isinstance(value, str):
            raise SchemaError(
                f"{path}.performance_environment[{key!r}] must be a string or null"
            )
    fingerprint_payload = dict(environment)
    observed_fingerprint = fingerprint_payload.pop("fingerprint")
    if canonical_fingerprint(fingerprint_payload) != observed_fingerprint:
        raise SchemaError(f"{path}.fingerprint does not match environment evidence")


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


def _validate_memory_measurement(value: Mapping[str, Any], path: str) -> None:
    _require_keys(
        value,
        ("matrix_bytes", "setup_bytes", "peak_estimate_bytes", "evidence"),
        path=path,
    )
    for field in ("matrix_bytes", "setup_bytes", "peak_estimate_bytes"):
        if value[field] is not None:
            _nonnegative_integer(value[field], f"{path}.{field}")
    _nonempty_string(value["evidence"], f"{path}.evidence")


def _certificate_kinds(capability: str, /) -> frozenset[str]:
    kinds = {
        "linear.scalar": {"linear-system"},
        "linear.block": {"linear-system"},
        "nonlinear.root": {"nonlinear-root"},
        "nonlinear.vi": {"variational-inequality-natural-map"},
        "eigen.general": {
            "eigenpair-relation",
            "schur-relation",
            "eigenpair-or-schur-relation",
        },
        "continuation.fold": {"continuation-branch-residual"},
        "optimization.unconstrained": {"optimization-stationarity"},
        "optimization.constrained": {"optimization-kkt"},
        "optimization.proximal": {"optimization-proximal-stationarity"},
        "optimization.bounded-least-squares": {"optimization-bound-stationarity"},
        "optimization.linear-program": {"optimization-program-kkt"},
        "optimization.quadratic-program": {"optimization-program-kkt"},
        "optimization.conic-program": {"optimization-program-kkt"},
        "optimization.mixed-integer-linear-program": {
            "mixed-integer-program-primal-global-reference"
        },
        "optimization.mixed-integer-conic-program": {
            "mixed-integer-program-primal-global-reference"
        },
    }
    try:
        return frozenset(kinds[capability])
    except KeyError as error:
        raise SchemaError(f"unknown certificate capability {capability!r}") from error


def _sha256(value: Any, path: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError(f"{path} must be a lowercase SHA-256 digest")


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
    "TIMING_PHASES",
    "SchemaError",
    "empty_distribution",
    "row_identity",
    "skip_certificate",
    "validate_report",
    "validate_row",
]
