#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Qualified isolated-root continuation with complete raw path accounting."""

from __future__ import annotations

import math
from enum import Enum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..backends.homotopy_continuation import (
    execute_homotopy_continuation,
    HomotopyContinuationExecution,
    HomotopyContinuationExecutionStatus,
    HomotopyContinuationPathStatus,
    HomotopyContinuationPolicy,
    HomotopyContinuationProvider,
    HomotopyContinuationRequest,
)
from ._grading import total_degree_bezout_forecast
from ._system import PolynomialScaling, SparsePolynomialSystem


class PolynomialPathStatus(str, Enum):
    REGULAR_ENDPOINT = "regular_endpoint"
    SINGULAR_ENDPOINT_CANDIDATE = "singular_endpoint_candidate"
    AT_INFINITY = "at_infinity"
    EXCESS_SOLUTION = "excess_solution"
    TRACKING_FAILED = "tracking_failed"
    INVALID_ENDPOINT = "invalid_endpoint"


class PolynomialSolveStatus(str, Enum):
    PROVIDER_FAILED = "provider_failed"
    INVALID_PROVIDER_OUTPUT = "invalid_provider_output"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    PATH_CAPACITY_EXCEEDED = "path_capacity_exceeded"
    PARTIAL_PATH_FAILURE = "partial_path_failure"
    RESIDUAL_REJECTED = "residual_rejected"
    SUCCESS = "success"


class IsolatedPolynomialRootProblem(StrictModule):
    """One square affine system and optional diagonal provider conditioning."""

    system: SparsePolynomialSystem
    scaling: PolynomialScaling | None
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: SparsePolynomialSystem,
        scaling: PolynomialScaling | None = None,
        /,
    ):
        if not isinstance(system, SparsePolynomialSystem):
            raise TypeError("system must be a SparsePolynomialSystem.")
        support = system.support
        if support.equation_count != support.variable_count:
            raise ValueError("Isolated affine root solving requires a square system.")
        if any(group.geometry != "affine" for group in support.groups):
            raise ValueError(
                "This isolated-root boundary accepts affine variable geometry only."
            )
        if scaling is not None:
            if not isinstance(scaling, PolynomialScaling):
                raise TypeError("scaling must be PolynomialScaling or None.")
            scaling._validate_system(system)
        self.system = system
        self.scaling = scaling
        self.problem_id = canonical_fingerprint(
            {
                "kind": "isolated-polynomial-root-problem",
                "system": system.system_id,
                "scaling": None if scaling is None else scaling.scaling_id,
            }
        )

    @property
    def provider_system(self) -> SparsePolynomialSystem:
        return (
            self.system
            if self.scaling is None
            else self.scaling.scale_system(self.system)
        )

    def to_physical_points(self, points: Array, /) -> Array:
        if self.scaling is None:
            return jnp.asarray(points)
        return self.scaling.to_physical_points(points)


class IsolatedRootPlan(StrictModule):
    problem: IsolatedPolynomialRootProblem
    provider: HomotopyContinuationProvider = eqx.field(static=True)
    policy: HomotopyContinuationPolicy = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    cluster_tolerance: float = eqx.field(static=True)
    near_real_absolute_tolerance: float = eqx.field(static=True)
    near_real_relative_tolerance: float = eqx.field(static=True)
    total_degree_path_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedIsolatedRootSolve(StrictModule):
    plan: IsolatedRootPlan
    provider_system: SparsePolynomialSystem
    request: HomotopyContinuationRequest = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)


class PolynomialPathEvidence(StrictModule):
    path_index: int = eqx.field(static=True)
    return_code: str = eqx.field(static=True)
    status: PolynomialPathStatus = eqx.field(static=True)
    endpoint: Array
    has_endpoint: bool = eqx.field(static=True)
    provider_residual_norm: float | None = eqx.field(static=True)
    original_residual_norm: float | None = eqx.field(static=True)
    near_real_imaginary_norm: float | None = eqx.field(static=True)
    near_real_threshold: float | None = eqx.field(static=True)
    near_real: bool = eqx.field(static=True)
    independently_accepted: bool = eqx.field(static=True)
    cluster_index: int | None = eqx.field(static=True)


class PolynomialCoverageEvidence(StrictModule):
    start_count: int = eqx.field(static=True)
    tracked_path_count: int = eqx.field(static=True)
    classified_path_count: int = eqx.field(static=True)
    regular_endpoint_count: int = eqx.field(static=True)
    singular_endpoint_candidate_count: int = eqx.field(static=True)
    at_infinity_count: int = eqx.field(static=True)
    excess_solution_count: int = eqx.field(static=True)
    tracking_failed_count: int = eqx.field(static=True)
    invalid_endpoint_count: int = eqx.field(static=True)
    independently_accepted_path_count: int = eqx.field(static=True)
    independently_rejected_path_count: int = eqx.field(static=True)
    cluster_count: int = eqx.field(static=True)
    all_tracked_paths_accounted: bool = eqx.field(static=True)
    scope: str = eqx.field(
        static=True,
        default="provider-start-path-accounting-not-proof-of-root-completeness",
    )


class PolynomialProviderEvidence(StrictModule):
    provider_id: str = eqx.field(static=True)
    environment_id: str = eqx.field(static=True)
    protocol_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    execution_status: str = eqx.field(static=True)
    raw_counts: tuple[tuple[str, int], ...] = eqx.field(static=True)
    raw_return_codes: tuple[str, ...] = eqx.field(static=True)
    start_count: int = eqx.field(static=True)
    tracked_path_count: int = eqx.field(static=True)
    process_returncode: int | None = eqx.field(static=True)
    process_timed_out: bool = eqx.field(static=True)
    run_artifact_id: str | None = eqx.field(static=True)
    error: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PolynomialRootResult(StrictModule):
    roots: Array
    root_mask: Array
    cluster_path_counts: Array
    near_real_mask: Array
    paths: tuple[PolynomialPathEvidence, ...]
    coverage: PolynomialCoverageEvidence
    provider: PolynomialProviderEvidence
    status: PolynomialSolveStatus = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    claim: str = eqx.field(
        static=True,
        default="numerically-replayed-finite-candidates-not-certified-all-roots",
    )

    @property
    def successful(self) -> Array:
        return jnp.asarray(self.status is PolynomialSolveStatus.SUCCESS)

    @property
    def root_count(self) -> Array:
        return jnp.sum(self.root_mask, dtype=jnp.int32)


def _tolerance(value: float, name: str, *, positive: bool) -> float:
    result = float(value)
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return result


def plan_isolated_roots(
    problem: IsolatedPolynomialRootProblem | SparsePolynomialSystem,
    provider: HomotopyContinuationProvider,
    /,
    *,
    policy: HomotopyContinuationPolicy | None = None,
    residual_tolerance: float = 1e-8,
    cluster_tolerance: float = 1e-7,
    near_real_absolute_tolerance: float = 1e-8,
    near_real_relative_tolerance: float = 1e-8,
) -> IsolatedRootPlan:
    """Plan a bounded provider solve; no provider is discovered or substituted."""

    problem_ = (
        IsolatedPolynomialRootProblem(problem)
        if isinstance(problem, SparsePolynomialSystem)
        else problem
    )
    if not isinstance(problem_, IsolatedPolynomialRootProblem):
        raise TypeError(
            "problem must be IsolatedPolynomialRootProblem or SparsePolynomialSystem."
        )
    if not isinstance(provider, HomotopyContinuationProvider):
        raise TypeError("provider must be HomotopyContinuationProvider.")
    policy_ = HomotopyContinuationPolicy() if policy is None else policy
    if not isinstance(policy_, HomotopyContinuationPolicy):
        raise TypeError("policy must be HomotopyContinuationPolicy.")
    residual = _tolerance(residual_tolerance, "residual_tolerance", positive=True)
    cluster = _tolerance(cluster_tolerance, "cluster_tolerance", positive=True)
    near_absolute = _tolerance(
        near_real_absolute_tolerance,
        "near_real_absolute_tolerance",
        positive=False,
    )
    near_relative = _tolerance(
        near_real_relative_tolerance,
        "near_real_relative_tolerance",
        positive=False,
    )
    forecast = total_degree_bezout_forecast(problem_.system.support).path_count
    identifier = canonical_fingerprint(
        {
            "kind": "isolated-root-plan",
            "problem": problem_.problem_id,
            "provider": provider.provider_id,
            "policy": policy_.policy_id,
            "residual_tolerance": residual,
            "cluster_tolerance": cluster,
            "near_real_absolute_tolerance": near_absolute,
            "near_real_relative_tolerance": near_relative,
            "total_degree_path_count": forecast,
        }
    )
    return IsolatedRootPlan(
        problem_,
        provider,
        policy_,
        residual,
        cluster,
        near_absolute,
        near_relative,
        forecast,
        identifier,
    )


def prepare_isolated_roots(plan: IsolatedRootPlan, /) -> PreparedIsolatedRootSolve:
    """Bind scaled numeric coefficients to a fixed support and provider plan."""

    if not isinstance(plan, IsolatedRootPlan):
        raise TypeError("plan must be IsolatedRootPlan.")
    system = plan.problem.provider_system
    support = system.support
    coefficients = np.asarray(system.coefficients)
    request_id = canonical_fingerprint(
        {
            "kind": "isolated-root-provider-request",
            "plan": plan.plan_id,
            "support": support.support_id,
            "system": system.system_id,
        }
    )
    request = HomotopyContinuationRequest(
        request_id,
        support.support_id,
        system.system_id,
        support.equation_count,
        support.variable_count,
        tuple(np.asarray(support.equation_indices)),
        tuple(tuple(row) for row in np.asarray(support.exponents)),
        tuple(complex(value) for value in coefficients),
    )
    preparation_id = canonical_fingerprint(
        {
            "kind": "prepared-isolated-root-solve",
            "plan": plan.plan_id,
            "request": request.request_id,
        }
    )
    return PreparedIsolatedRootSolve(
        plan,
        system,
        request,
        support.support_id,
        preparation_id,
    )


def refresh_isolated_roots(
    prepared: PreparedIsolatedRootSolve,
    problem: IsolatedPolynomialRootProblem | SparsePolynomialSystem,
    /,
) -> PreparedIsolatedRootSolve:
    """Refresh coefficients/scales only; reject every support structural mismatch."""

    if not isinstance(prepared, PreparedIsolatedRootSolve):
        raise TypeError("prepared must be PreparedIsolatedRootSolve.")
    problem_ = (
        IsolatedPolynomialRootProblem(problem)
        if isinstance(problem, SparsePolynomialSystem)
        else problem
    )
    if not isinstance(problem_, IsolatedPolynomialRootProblem):
        raise TypeError(
            "problem must be IsolatedPolynomialRootProblem or SparsePolynomialSystem."
        )
    if problem_.system.support.support_id != prepared.support_id:
        raise ValueError(
            "Cannot refresh an isolated solve across a structural support mismatch."
        )
    old = prepared.plan
    plan = plan_isolated_roots(
        problem_,
        old.provider,
        policy=old.policy,
        residual_tolerance=old.residual_tolerance,
        cluster_tolerance=old.cluster_tolerance,
        near_real_absolute_tolerance=old.near_real_absolute_tolerance,
        near_real_relative_tolerance=old.near_real_relative_tolerance,
    )
    return prepare_isolated_roots(plan)


def _provider_evidence(
    prepared: PreparedIsolatedRootSolve,
    execution: HomotopyContinuationExecution | None,
    /,
    *,
    execution_status: str = "path_capacity_exceeded_before_provider",
    start_count: int = 0,
) -> PolynomialProviderEvidence:
    provider = prepared.plan.provider
    policy = prepared.plan.policy
    if execution is None:
        raw_counts: tuple[tuple[str, int], ...] = ()
        return_codes: tuple[str, ...] = ()
        tracked = 0
        run = None
        error = "Declared total-degree start count exceeds path_capacity."
    else:
        execution_status = execution.status.value
        raw_counts = execution.counts
        return_codes = tuple(path.return_code for path in execution.paths)
        start_count = execution.start_count
        tracked = execution.tracked_path_count
        run = execution.run
        error = execution.error
    evidence_id = canonical_fingerprint(
        {
            "kind": "polynomial-provider-evidence",
            "provider": provider.provider_id,
            "environment": provider.environment.environment_id,
            "protocol": provider.protocol_id,
            "policy": policy.policy_id,
            "execution_status": execution_status,
            "raw_counts": list(raw_counts),
            "raw_return_codes": list(return_codes),
            "start_count": start_count,
            "tracked_path_count": tracked,
            "run": None if run is None else run.artifact.artifact_id,
            "error": error,
        }
    )
    return PolynomialProviderEvidence(
        provider.provider_id,
        provider.environment.environment_id,
        provider.protocol_id,
        policy.policy_id,
        execution_status,
        raw_counts,
        return_codes,
        int(start_count),
        int(tracked),
        None if run is None else run.returncode,
        False if run is None else run.timed_out,
        None if run is None else run.artifact.artifact_id,
        error,
        evidence_id,
    )


def _empty_coverage(start_count: int = 0) -> PolynomialCoverageEvidence:
    return PolynomialCoverageEvidence(
        int(start_count), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False
    )


def _empty_result(
    prepared: PreparedIsolatedRootSolve,
    status: PolynomialSolveStatus,
    provider: PolynomialProviderEvidence,
    coverage: PolynomialCoverageEvidence,
    /,
) -> PolynomialRootResult:
    capacity = prepared.plan.policy.path_capacity
    variables = prepared.request.variable_count
    roots = jnp.full((capacity, variables), complex(np.nan, np.nan), dtype=jnp.complex128)
    root_mask = jnp.zeros((capacity,), dtype=jnp.bool_)
    cluster_path_counts = jnp.zeros((capacity,), dtype=jnp.int32)
    near_real = jnp.zeros((capacity,), dtype=jnp.bool_)
    identifier = canonical_fingerprint(
        {
            "kind": "polynomial-root-result",
            "preparation": prepared.preparation_id,
            "status": status.value,
            "provider_evidence": provider.evidence_id,
            "accepted_roots": [],
        }
    )
    return PolynomialRootResult(
        roots,
        root_mask,
        cluster_path_counts,
        near_real,
        (),
        coverage,
        provider,
        status,
        prepared.plan.problem.problem_id,
        prepared.plan.plan_id,
        prepared.preparation_id,
        identifier,
    )


def _path_status(status: HomotopyContinuationPathStatus, /) -> PolynomialPathStatus:
    return PolynomialPathStatus(status.value)


def _endpoint_key(endpoint: np.ndarray, /) -> tuple[float, ...]:
    return tuple(
        component
        for value in endpoint
        for component in (float(value.real), float(value.imag))
    )


def _cluster(
    candidates: list[tuple[int, np.ndarray, float, bool]],
    tolerance: float,
    /,
) -> tuple[list[tuple[np.ndarray, int, bool, tuple[int, ...]]], dict[int, int]]:
    """Deterministic connected components under a symmetric scaled distance gate."""

    ordered = sorted(candidates, key=lambda item: (_endpoint_key(item[1]), item[0]))
    count = len(ordered)
    parent = list(range(count))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(count):
        left_endpoint = ordered[left][1]
        for right in range(left + 1, count):
            right_endpoint = ordered[right][1]
            scale = max(
                1.0,
                float(np.max(np.abs(left_endpoint), initial=0.0)),
                float(np.max(np.abs(right_endpoint), initial=0.0)),
            )
            distance = float(np.max(np.abs(left_endpoint - right_endpoint), initial=0.0))
            if distance <= tolerance * scale:
                union(left, right)
    components: dict[int, list[tuple[int, np.ndarray, float, bool]]] = {}
    for index, candidate in enumerate(ordered):
        components.setdefault(root(index), []).append(candidate)
    selected = []
    for members in components.values():
        representative = min(
            members,
            key=lambda item: (item[2], _endpoint_key(item[1]), item[0]),
        )
        selected.append(
            (
                representative[1],
                len(members),
                representative[3],
                tuple(sorted(member[0] for member in members)),
            )
        )
    selected.sort(key=lambda item: (_endpoint_key(item[0]), item[3]))
    mapping = {
        path_index: cluster_index
        for cluster_index, (_, _, _, members) in enumerate(selected)
        for path_index in members
    }
    return selected, mapping


def solve_prepared_isolated_roots(
    prepared: PreparedIsolatedRootSolve, /
) -> PolynomialRootResult:
    """Track, independently replay, cluster and retain every raw path disposition."""

    if not isinstance(prepared, PreparedIsolatedRootSolve):
        raise TypeError("prepared must be PreparedIsolatedRootSolve.")
    plan = prepared.plan
    if (
        plan.policy.start_system == "total-degree"
        and plan.total_degree_path_count > plan.policy.path_capacity
    ):
        provider = _provider_evidence(
            prepared, None, start_count=plan.total_degree_path_count
        )
        return _empty_result(
            prepared,
            PolynomialSolveStatus.PATH_CAPACITY_EXCEEDED,
            provider,
            _empty_coverage(plan.total_degree_path_count),
        )
    execution = execute_homotopy_continuation(
        plan.provider, plan.policy, prepared.request
    )
    provider = _provider_evidence(prepared, execution)
    failure_status = {
        HomotopyContinuationExecutionStatus.PROVIDER_FAILED: PolynomialSolveStatus.PROVIDER_FAILED,
        HomotopyContinuationExecutionStatus.INVALID_OUTPUT: PolynomialSolveStatus.INVALID_PROVIDER_OUTPUT,
        HomotopyContinuationExecutionStatus.SEMANTIC_MISMATCH: PolynomialSolveStatus.SEMANTIC_MISMATCH,
        HomotopyContinuationExecutionStatus.PATH_CAPACITY_EXCEEDED: PolynomialSolveStatus.PATH_CAPACITY_EXCEEDED,
    }
    if execution.status in failure_status:
        return _empty_result(
            prepared,
            failure_status[execution.status],
            provider,
            _empty_coverage(execution.start_count),
        )
    original_system = plan.problem.system
    variables = prepared.request.variable_count
    blank = np.full((variables,), complex(np.nan, np.nan))
    provisional = []
    candidates: list[tuple[int, np.ndarray, float, bool]] = []
    rejected = 0
    for path in execution.paths:
        status = _path_status(path.status)
        endpoint = blank
        has_endpoint = path.endpoint is not None
        original_residual = None
        imaginary_norm = None
        near_threshold = None
        near_real = False
        accepted = False
        if has_endpoint:
            scaled = jnp.asarray(path.endpoint)
            physical = np.asarray(
                plan.problem.to_physical_points(scaled), dtype=np.complex128
            )
            endpoint = physical
            residual = np.asarray(original_system.evaluate(physical))
            original_residual = float(np.max(np.abs(residual), initial=0.0))
            real_scale = float(np.max(np.abs(physical.real), initial=0.0))
            imaginary_norm = float(np.max(np.abs(physical.imag), initial=0.0))
            near_threshold = (
                plan.near_real_absolute_tolerance
                + plan.near_real_relative_tolerance * max(1.0, real_scale)
            )
            near_real = imaginary_norm <= near_threshold
            accepted = bool(
                np.all(np.isfinite(physical))
                and math.isfinite(original_residual)
                and original_residual <= plan.residual_tolerance
            )
            if accepted:
                candidates.append(
                    (path.path_index, physical, original_residual, near_real)
                )
            else:
                rejected += 1
        provisional.append(
            (
                path,
                status,
                endpoint,
                has_endpoint,
                original_residual,
                imaginary_norm,
                near_threshold,
                near_real,
                accepted,
            )
        )
    clusters, cluster_mapping = _cluster(candidates, plan.cluster_tolerance)
    path_evidence = tuple(
        PolynomialPathEvidence(
            path.path_index,
            path.return_code,
            status,
            jnp.asarray(endpoint),
            has_endpoint,
            path.provider_residual_norm,
            original_residual,
            imaginary_norm,
            near_threshold,
            near_real,
            accepted,
            cluster_mapping.get(path.path_index),
        )
        for (
            path,
            status,
            endpoint,
            has_endpoint,
            original_residual,
            imaginary_norm,
            near_threshold,
            near_real,
            accepted,
        ) in provisional
    )
    capacity = plan.policy.path_capacity
    roots_host = np.full((capacity, variables), complex(np.nan, np.nan))
    mask_host = np.zeros((capacity,), dtype=np.bool_)
    cluster_path_counts_host = np.zeros((capacity,), dtype=np.int32)
    near_real_host = np.zeros((capacity,), dtype=np.bool_)
    for index, (root, cluster_path_count, near_real, _) in enumerate(clusters):
        roots_host[index] = root
        mask_host[index] = True
        cluster_path_counts_host[index] = cluster_path_count
        near_real_host[index] = near_real
    counts = dict(execution.counts)
    failed = counts.get(PolynomialPathStatus.TRACKING_FAILED.value, 0)
    invalid = counts.get(PolynomialPathStatus.INVALID_ENDPOINT.value, 0)
    coverage = PolynomialCoverageEvidence(
        execution.start_count,
        execution.tracked_path_count,
        sum(counts.values()),
        counts.get(PolynomialPathStatus.REGULAR_ENDPOINT.value, 0),
        counts.get(PolynomialPathStatus.SINGULAR_ENDPOINT_CANDIDATE.value, 0),
        counts.get(PolynomialPathStatus.AT_INFINITY.value, 0),
        counts.get(PolynomialPathStatus.EXCESS_SOLUTION.value, 0),
        failed,
        invalid,
        len(candidates),
        rejected,
        len(clusters),
        execution.start_count == execution.tracked_path_count == sum(counts.values()),
    )
    if failed or invalid:
        status = PolynomialSolveStatus.PARTIAL_PATH_FAILURE
    elif rejected:
        status = PolynomialSolveStatus.RESIDUAL_REJECTED
    else:
        status = PolynomialSolveStatus.SUCCESS
    identifier = canonical_fingerprint(
        {
            "kind": "polynomial-root-result",
            "preparation": prepared.preparation_id,
            "status": status.value,
            "provider_evidence": provider.evidence_id,
            "accepted_roots": [
                {
                    "endpoint": [[value.real, value.imag] for value in root],
                    "cluster_path_count": cluster_path_count,
                    "near_real": near_real,
                    "paths": list(paths),
                }
                for root, cluster_path_count, near_real, paths in clusters
            ],
        }
    )
    return PolynomialRootResult(
        jnp.asarray(roots_host),
        jnp.asarray(mask_host),
        jnp.asarray(cluster_path_counts_host),
        jnp.asarray(near_real_host),
        path_evidence,
        coverage,
        provider,
        status,
        plan.problem.problem_id,
        plan.plan_id,
        prepared.preparation_id,
        identifier,
    )


__all__ = [
    "IsolatedPolynomialRootProblem",
    "PolynomialPathStatus",
    "PolynomialSolveStatus",
    "PolynomialPathEvidence",
    "PolynomialCoverageEvidence",
    "PolynomialProviderEvidence",
    "PolynomialRootResult",
    "IsolatedRootPlan",
    "PreparedIsolatedRootSolve",
    "plan_isolated_roots",
    "prepare_isolated_roots",
    "refresh_isolated_roots",
    "solve_prepared_isolated_roots",
]
