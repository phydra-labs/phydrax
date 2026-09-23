#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks._runtime import (
    capture_environment,
    DurationDistribution,
    measure_repeated,
    measure_synchronized,
)

from .adapters.base import (
    Availability,
    BenchmarkAdapter,
    CaseSpec,
    NOT_APPLICABLE_REFRESH,
    RefreshEvidence,
    SKIPPED_TRANSFERS,
    SolveResult,
)
from .certificates import certificate_evaluator_fingerprint, independent_certificate
from .schema import (
    empty_distribution,
    skip_certificate,
    TIMING_PHASES,
    validate_report,
)


def execute_case(
    adapter: BenchmarkAdapter,
    spec: CaseSpec,
    /,
    *,
    environment: Mapping[str, Any],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Execute one adapter/case pair and preserve phase-specific failure evidence."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    timing = {phase: empty_distribution() for phase in TIMING_PHASES}
    phase = "availability"
    availability: Availability | None = None
    implementation = {
        "adapter": adapter.name,
        "backend": adapter.dependency,
        "method": "not-initialized",
        "preconditioner": "none",
        "versions": {},
    }
    prepared_state: Any = None
    try:
        availability = adapter.availability(spec.capability)
        phase = "implementation"
        implementation = adapter.implementation(spec).as_dict()
        if not availability.available:
            return _skip_row(
                spec,
                implementation=implementation,
                availability=availability,
                environment=environment,
            )

        phase = "setup"
        setup_state, sample = measure_synchronized(lambda: adapter.setup(spec))
        prepared_state = setup_state
        timing["setup"] = DurationDistribution((sample,)).to_milliseconds_dict()

        compilation_applicable = adapter.compilation_applicable(setup_state)
        compile_after_preparation = (
            compilation_applicable and adapter.compilation_after_preparation(setup_state)
        )
        compiled_state = setup_state
        if compilation_applicable and not compile_after_preparation:
            phase = "compilation"
            compiled_state, sample = measure_synchronized(
                lambda: adapter.compile(setup_state)
            )
            prepared_state = compiled_state
            timing["compilation"] = DurationDistribution(
                (sample,)
            ).to_milliseconds_dict()

        if adapter.preparation_applicable(compiled_state):
            phase = "preparation"
            prepared_state, sample = measure_synchronized(
                lambda: adapter.prepare(compiled_state)
            )
            timing["preparation"] = DurationDistribution(
                (sample,)
            ).to_milliseconds_dict()
        else:
            prepared_state = compiled_state

        if compile_after_preparation:
            phase = "compilation"
            prepared_state, sample = measure_synchronized(
                lambda: adapter.compile(prepared_state)
            )
            timing["compilation"] = DurationDistribution(
                (sample,)
            ).to_milliseconds_dict()
        adapter.mark_warmup(prepared_state, warmup)

        if warmup:
            phase = "warmup"
            _, warmup_distribution = measure_repeated(
                lambda: adapter.solve(prepared_state),
                warmup=0,
                repeats=warmup,
            )
            timing["warmup"] = warmup_distribution.to_milliseconds_dict()

        phase = "solve"
        result, solve_distribution = measure_repeated(
            lambda: adapter.solve(prepared_state),
            warmup=0,
            repeats=repeats,
        )
        timing["solve"] = solve_distribution.to_milliseconds_dict()

        if adapter.differentiation_applicable(prepared_state):
            phase = "differentiation_compilation"
            prepared_state, sample = measure_synchronized(
                lambda: adapter.compile_differentiation(prepared_state)
            )
            timing["differentiation_compilation"] = DurationDistribution(
                (sample,)
            ).to_milliseconds_dict()
            phase = "differentiation"
            _, differentiation_distribution = measure_repeated(
                lambda: adapter.differentiate(prepared_state),
                warmup=0,
                repeats=repeats,
            )
            timing["differentiation"] = (
                differentiation_distribution.to_milliseconds_dict()
            )

        phase = "verification"
        certificate_problem = adapter.certificate_problem(prepared_state)
        verification, sample = measure_synchronized(
            lambda: _materialize_and_certify(
                adapter,
                prepared_state,
                certificate_problem,
                result,
                spec.tolerances,
            )
        )
        certificate, converged, operations = verification
        timing["verification"] = DurationDistribution(
            (sample,)
        ).to_milliseconds_dict()
        initial_memory = adapter.memory(prepared_state, result)

        refresh_evidence: RefreshEvidence = NOT_APPLICABLE_REFRESH
        refreshed_result: SolveResult | None = None
        refreshed_certificate_problem = None
        refreshed_certificate = None
        refreshed_converged = None
        refreshed_memory = None
        if adapter.refresh_applicable(prepared_state):
            phase = "refresh"
            refresh_result, refresh_sample = measure_synchronized(
                lambda: adapter.refresh(prepared_state)
            )
            prepared_state, refresh_evidence = refresh_result
            timing["refresh"] = DurationDistribution(
                (refresh_sample,)
            ).to_milliseconds_dict()
            phase = "refreshed_solve"
            refreshed_result, refreshed_solve_sample = measure_synchronized(
                lambda: adapter.solve(prepared_state)
            )
            timing["refreshed_solve"] = DurationDistribution(
                (refreshed_solve_sample,)
            ).to_milliseconds_dict()
            phase = "refreshed_verification"
            refreshed_certificate_problem = adapter.certificate_problem(prepared_state)
            expected_refreshed_problem = _expected_refreshed_problem(spec.problem)
            if (
                refreshed_certificate_problem.identity()["fingerprint"]
                != expected_refreshed_problem.identity()["fingerprint"]
            ):
                raise ValueError(
                    "adapter refreshed a problem other than the declared deterministic "
                    "numeric perturbation"
                )
            refreshed_verification, refreshed_verification_sample = (
                measure_synchronized(
                    lambda: _materialize_and_certify(
                        adapter,
                        prepared_state,
                        refreshed_certificate_problem,
                        refreshed_result,
                        spec.tolerances,
                    )
                )
            )
            refreshed_certificate, refreshed_converged, _ = refreshed_verification
            timing["refreshed_verification"] = DurationDistribution(
                (refreshed_verification_sample,)
            ).to_milliseconds_dict()
            refreshed_memory = adapter.memory(prepared_state, refreshed_result)

        phase = "transfer_accounting"
        transfer_results = [result]
        if refreshed_result is not None:
            transfer_results.append(refreshed_result)
        device_to_host_bytes = _device_array_bytes(
            tuple(
                (
                    measured_result.solution,
                    measured_result.auxiliary,
                    measured_result.converged,
                    measured_result.operations,
                )
                for measured_result in transfer_results
            )
        )
        transfers = adapter.transfers(
            prepared_state,
            result,
            device_to_host_bytes=device_to_host_bytes,
        )
        refresh_record = {
            **refresh_evidence.as_dict(),
            "certificate_problem_fingerprint": (
                None
                if refreshed_certificate_problem is None
                else refreshed_certificate_problem.identity()["fingerprint"]
            ),
            "certificate_kind": (
                None if refreshed_certificate is None else refreshed_certificate["kind"]
            ),
            "certificate_relative_residual": (
                None
                if refreshed_certificate is None
                else refreshed_certificate["relative_residual"]
            ),
            "certificate_backward_error": (
                None
                if refreshed_certificate is None
                else refreshed_certificate["backward_error"]
            ),
            "certificate_converged": refreshed_converged,
            "independently_certified": (
                None
                if refreshed_certificate is None
                else refreshed_certificate["independently_computed"]
            ),
        }
        lifecycle_converged = converged and (
            not refresh_evidence.applicable or refreshed_converged is True
        )
        status = "success" if lifecycle_converged else "nonconverged"
        row = {
            "case_id": spec.name,
            "environment": dict(environment),
            "problem": spec.problem.identity(),
            "implementation": implementation,
            "sizes": spec.problem.sizes(),
            "tolerances": spec.tolerances.as_dict(),
            "outcome": {
                "status": status,
                "converged": lifecycle_converged,
                "message": result.message,
                "skip_reason": None,
                "failure_phase": None,
            },
            "certificate": certificate,
            "operations": _operation_evidence(operations),
            "refresh": refresh_record,
            "memory": {
                "initial": initial_memory,
                "refreshed": refreshed_memory,
                "evidence": (
                    "initial memory was captured after initial verification and before "
                    "refresh; refreshed memory was captured after refreshed verification"
                ),
            },
            "transfers": transfers.as_dict(),
            "timing": timing,
            "availability": availability.as_dict(),
        }
        phase = "release"
        adapter.release(prepared_state)
        prepared_state = None
        return row
    except Exception as error:
        if availability is None:
            availability = Availability(
                available=False,
                capability=spec.capability,
                dependency=adapter.dependency,
                dependency_version=None,
                reason=(
                    f"provider initialization failed during {phase}: "
                    f"{type(error).__name__}: {error}"
                ),
            )
        return _failure_row(
            spec,
            implementation=implementation,
            availability=availability,
            environment=environment,
            phase=phase,
            error=error,
            timing=timing,
        )
    finally:
        if prepared_state is not None:
            try:
                adapter.release(prepared_state)
            except Exception:
                pass


def run_campaign(
    adapters: Mapping[str, BenchmarkAdapter],
    config: Any,
    /,
) -> dict[str, Any]:
    """Build and run one immutable deterministic campaign descriptor."""
    import jax

    from .campaign import build_cases

    jax.config.update("jax_enable_x64", True)
    selected_adapters = tuple(config.adapters)
    selected_cases = tuple(config.cases)
    cases = build_cases(config)
    unknown_adapters = sorted(set(selected_adapters) - adapters.keys())
    unknown_cases = sorted(set(selected_cases) - cases.keys())
    if unknown_adapters:
        raise ValueError(f"unknown adapters: {', '.join(unknown_adapters)}")
    if unknown_cases:
        raise ValueError(f"unknown cases: {', '.join(unknown_cases)}")
    if len(set(selected_adapters)) != len(selected_adapters):
        raise ValueError("selected_adapters must not contain duplicates")
    if len(set(selected_cases)) != len(selected_cases):
        raise ValueError("selected_cases must not contain duplicates")
    environment = capture_environment().to_dict()
    rows = [
        execute_case(
            adapters[adapter_name],
            cases[case_name],
            environment=environment,
            warmup=config.warmup,
            repeats=config.repeats,
        )
        for case_name in selected_cases
        for adapter_name in selected_adapters
    ]
    passed = all(
        row["outcome"]["status"] in {"success", "skipped"} for row in rows
    )
    report = {
        "passed": passed,
        "environment": environment,
        "provenance": {
            "harness_source_fingerprint": _source_fingerprint(Path(__file__)),
            "certificate_evaluator_fingerprint": (
                certificate_evaluator_fingerprint()
            ),
            "case_source_fingerprint": _source_fingerprint(
                Path(__file__).with_name("problems.py")
            ),
        },
        "campaign": {
            "seed": config.seed,
            "warmup": config.warmup,
            "repeats": config.repeats,
            "selected_adapters": list(selected_adapters),
            "selected_cases": list(selected_cases),
            "case_fingerprints": {
                name: cases[name].problem.identity()["fingerprint"]
                for name in selected_cases
            },
            "python_executable": sys.executable,
        },
        "rows": rows,
    }
    validate_report(report)
    return report


def _skip_row(
    spec: CaseSpec,
    *,
    implementation: dict[str, Any],
    availability: Availability,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    if availability.reason is None or not availability.reason.strip():
        raise ValueError("unavailable adapter must provide a precise skip reason")
    return {
        "case_id": spec.name,
        "environment": dict(environment),
        "problem": spec.problem.identity(),
        "implementation": implementation,
        "sizes": spec.problem.sizes(),
        "tolerances": spec.tolerances.as_dict(),
        "outcome": {
            "status": "skipped",
            "converged": None,
            "message": "not executed",
            "skip_reason": availability.reason,
            "failure_phase": None,
        },
        "certificate": skip_certificate(
            _certificate_kind(spec.capability),
            capability=spec.capability,
            problem_fingerprint=spec.problem.identity()["fingerprint"],
        ),
        "operations": _operation_evidence({}),
        "refresh": {
            **NOT_APPLICABLE_REFRESH.as_dict(),
            "certificate_problem_fingerprint": None,
            "certificate_kind": None,
            "certificate_relative_residual": None,
            "certificate_backward_error": None,
            "certificate_converged": None,
            "independently_certified": None,
        },
        "memory": {
            "initial": None,
            "refreshed": None,
            "evidence": "not measured because the row was skipped",
        },
        "transfers": SKIPPED_TRANSFERS.as_dict(),
        "timing": {phase: empty_distribution() for phase in TIMING_PHASES},
        "availability": availability.as_dict(),
    }


def _failure_row(
    spec: CaseSpec,
    *,
    implementation: dict[str, Any],
    availability: Availability,
    environment: Mapping[str, Any],
    phase: str,
    error: Exception,
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    transfers = SKIPPED_TRANSFERS.as_dict()
    transfers["evidence"] = (
        f"not measured because provider execution failed during {phase}"
    )
    return {
        "case_id": spec.name,
        "environment": dict(environment),
        "problem": spec.problem.identity(),
        "implementation": implementation,
        "sizes": spec.problem.sizes(),
        "tolerances": spec.tolerances.as_dict(),
        "outcome": {
            "status": "failed",
            "converged": False,
            "message": f"{type(error).__name__}: {error}",
            "skip_reason": None,
            "failure_phase": phase,
        },
        "certificate": skip_certificate(
            _certificate_kind(spec.capability),
            capability=spec.capability,
            problem_fingerprint=spec.problem.identity()["fingerprint"],
        ),
        "operations": _operation_evidence({}),
        "refresh": {
            **NOT_APPLICABLE_REFRESH.as_dict(),
            "certificate_problem_fingerprint": None,
            "certificate_kind": None,
            "certificate_relative_residual": None,
            "certificate_backward_error": None,
            "certificate_converged": None,
            "independently_certified": None,
        },
        "memory": {
            "initial": None,
            "refreshed": None,
            "evidence": f"not measured because provider execution failed during {phase}",
        },
        "transfers": transfers,
        "timing": dict(timing),
        "availability": availability.as_dict(),
    }


def _operation_evidence(values: Mapping[str, Any | None]) -> dict[str, int | None]:
    operation_fields = (
        "iterations",
        "matvecs",
        "preconditioner_applications",
        "linear_solves",
        "nonlinear_evaluations",
        "jacobian_evaluations",
    )
    evidence: dict[str, int | None] = {}
    for field in operation_fields:
        value = values.get(field)
        evidence[field] = None if value is None else int(np.asarray(value))
    return evidence


def _materialize_and_certify(
    adapter: BenchmarkAdapter,
    prepared_state: Any,
    problem: Any,
    result: SolveResult,
    tolerances: Any,
    /,
) -> tuple[dict[str, Any], bool, dict[str, Any | None]]:
    solution, auxiliary, converged, operations = adapter.materialize_result(
        prepared_state,
        result,
    )
    converged_array = np.asarray(converged)
    if converged_array.shape != () or converged_array.dtype.kind != "b":
        raise TypeError("adapter convergence evidence must be one boolean scalar")
    certificate = independent_certificate(problem, solution, auxiliary)
    certificate["problem_fingerprint"] = problem.identity()["fingerprint"]
    certificate["capability"] = problem.capability
    certificate["evaluator_fingerprint"] = certificate_evaluator_fingerprint()
    effective_converged = bool(converged_array)
    tolerance_satisfied = (
        certificate["residual_norm"] <= tolerances.absolute
        or certificate["relative_residual"] <= tolerances.relative
    )
    effective_converged = effective_converged and tolerance_satisfied
    from .problems import ContinuationProblem, GeneralEigenProblem

    if isinstance(problem, ContinuationProblem):
        effective_converged = (
            effective_converged and certificate["details"]["successful_fold_traversal"]
        )
    if isinstance(problem, GeneralEigenProblem):
        effective_converged = (
            effective_converged
            and certificate["details"]["count_satisfied"]
            and certificate["details"]["largest_magnitude_membership_satisfied"]
        )
    return certificate, effective_converged, operations


def _source_fingerprint(path: Path, /) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_refreshed_problem(problem: Any, /) -> Any:
    from .problems import (
        ContinuationProblem,
        GeneralEigenProblem,
        MathematicalProgramProblem,
        NonlinearProblem,
        SparseLinearProblem,
    )

    if isinstance(problem, SparseLinearProblem):
        return replace(problem, coefficients=problem.coefficients * 1.01)
    if isinstance(problem, GeneralEigenProblem):
        return replace(problem, matrix=problem.matrix * 1.01)
    if isinstance(problem, NonlinearProblem) and problem.variant == "root":
        return replace(problem, target=problem.target * 1.01)
    if isinstance(problem, MathematicalProgramProblem):
        return replace(
            problem,
            quadratic=(
                None if problem.quadratic is None else problem.quadratic * 1.01
            ),
            linear=problem.linear * 1.01,
        )
    if isinstance(problem, ContinuationProblem):
        return replace(
            problem,
            initial_coordinate=problem.initial_coordinate * 0.99,
        )
    raise TypeError(
        f"no deterministic refresh contract for {type(problem).__name__!r}"
    )


def _device_array_bytes(value: Any, /) -> int:
    import jax

    seen: set[int] = set()
    total = 0

    def visit(item: Any) -> None:
        nonlocal total
        if isinstance(item, jax.Array):
            identifier = id(item)
            if identifier not in seen:
                seen.add(identifier)
                total += item.size * item.dtype.itemsize
            return
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
            return
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(object.__getattribute__(item, field.name))
            return
        leaves = jax.tree.leaves(item)
        if len(leaves) == 1 and leaves[0] is item:
            return
        for leaf in leaves:
            visit(leaf)

    visit(value)
    return total


def _certificate_kind(capability: str) -> str:
    return {
        "linear.scalar": "linear-system",
        "linear.block": "linear-system",
        "nonlinear.root": "nonlinear-root",
        "nonlinear.vi": "variational-inequality-natural-map",
        "eigen.general": "eigenpair-or-schur-relation",
        "continuation.fold": "continuation-branch-residual",
        "optimization.unconstrained": "optimization-stationarity",
        "optimization.constrained": "optimization-kkt",
        "optimization.proximal": "optimization-proximal-stationarity",
        "optimization.bounded-least-squares": ("optimization-bound-stationarity"),
        "optimization.linear-program": "optimization-program-kkt",
        "optimization.quadratic-program": "optimization-program-kkt",
        "optimization.conic-program": "optimization-program-kkt",
        "optimization.mixed-integer-linear-program": (
            "mixed-integer-program-primal-global-reference"
        ),
        "optimization.mixed-integer-conic-program": (
            "mixed-integer-program-primal-global-reference"
        ),
    }[capability]


__all__ = ["execute_case", "run_campaign"]
