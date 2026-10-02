# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Phase-separated singular-subspace scaling, qualification, and runtime smoke."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, final, Literal, NoReturn

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import PyTree

from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
    synchronize,
)
from phydrax import linalg as la
from phydrax.linalg._svd_qualification import (
    singular_subspace_candidate_profiles,
    singular_subspace_qualification_record,
    SingularSubspaceScenario,
)
from phydrax.linalg.svd import (
    DenseSVD,
    plan_svd,
    prepare_svd,
    PreparedSVDSolve,
    projector_action,
    RandomizedSVD,
    refresh_svd,
    svd,
    SVDApproximationPolicy,
    SVDDifferentiationMode,
    SVDProblem,
    SVDResourcePolicy,
    SVDSolvePlan,
    SVDSolvePolicy,
    SVDSolveResult,
    SVDTolerancePolicy,
)
from phydrax.ml import MLBatch
from phydrax.ml.decomposition import (
    IncrementalPCA,
    IncrementalPCAModel,
    PCA,
    POD,
    SubspaceModel,
)
from phydrax.nn.operator import OperatorAxis
from phydrax.nn.operator.training import (
    fit_operator_pod,
    operator_dataset_from_arrays,
)
from phydrax.qualification._core_portfolio import CoreQualificationObservation
from phydrax.qualification._registry import CapabilityProfile
from phydrax.typing import (
    as_host_array,
    Dim,
    HostFloat64,
    HostInexact,
    parse,
    PRNGKey,
    Scope,
)


type Spectrum = Literal["separated", "slow", "boundary", "low-rank"]


class TargetDim(Dim):
    pass


class SourceDim(Dim):
    pass


class SingularDim(Dim):
    pass


@dataclass(frozen=True, slots=True)
class CampaignCase:
    rows: int
    columns: int
    rank: int
    dtype: str = "float64"
    spectrum: Spectrum = "separated"
    randomized: bool = False
    oversampling: int = 8
    power_iterations: int = 1

    @property
    def identifier(self) -> str:
        route = "randomized" if self.randomized else "dense"
        return f"{route}/{self.rows}x{self.columns}/k{self.rank}/{self.dtype}/{self.spectrum}/p{self.oversampling}/q{self.power_iterations}"


@final
class NonmaterializingBenchmarkOperator(la.AbstractLinearOperator):
    """Enforce an action-only provider boundary over native low-rank algebra."""

    operator: la.AbstractLinearOperator
    _fused_block_action_kind: ClassVar[str] = "generic"

    def __init__(self, operator: la.AbstractLinearOperator, /) -> None:
        if not isinstance(operator, la.AbstractLinearOperator):
            raise TypeError("The shell requires a native linear operator.")
        if operator.batch_shape:
            raise ValueError("The shell campaign requires an unbatched operator.")
        self.operator = operator
        self.source = operator.source
        self.target = operator.target
        self.properties = operator.properties
        self.capabilities = la.OperatorCapabilities(
            transpose=True, adjoint=True, materialize=False
        )
        self.batch_shape = ()
        self.operator_id = f"{operator.operator_id}:action-only"

    def mv(self, vector: PyTree[Array], /) -> PyTree[Array]:
        return self.operator.mv(vector)

    def transpose_mv(self, vector: PyTree[Array], /) -> PyTree[Array]:
        return self.operator.transpose_mv(vector)

    def adjoint_mv(self, vector: PyTree[Array], /) -> PyTree[Array]:
        return self.operator.adjoint_mv(vector)

    def _materialize(self, /) -> NoReturn:
        raise la.LinearCapabilityError("The benchmark provider forbids materialization.")


@jax.jit
def _prepare_entry(
    problem: SVDProblem, plan: SVDSolvePlan, key: PRNGKey | None
) -> PreparedSVDSolve:
    return prepare_svd(problem, plan, key=key)


@jax.jit
def _solve_entry(prepared: PreparedSVDSolve) -> SVDSolveResult:
    return svd(prepared)


@jax.jit
def _refresh_entry(prepared: PreparedSVDSolve, problem: SVDProblem) -> PreparedSVDSolve:
    return refresh_svd(prepared, problem)


@jax.jit
def _projection_entry(
    result: SVDSolveResult, source: la.ArraySpace, probe: Array
) -> Array:
    return projector_action(
        result.right_coordinates, result.right_response, source.riesz(probe)
    )


def _compile_phase(
    lower: Callable[[], jax.stages.Lowered],
) -> tuple[jax.stages.Compiled, dict[str, Any]]:
    executable, timing = measure_lower_and_compile(
        lower, lambda lowered: lowered.compile()
    )
    evidence = compiler_evidence(
        executable.cost_analysis(),
        executable.memory_analysis(),
        source="jax-aot",
    )
    return executable, {
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "compiler": asdict(evidence),
    }


def _timed[T](operation: Callable[[], T], repeats: int) -> tuple[T, dict[str, Any]]:
    value, durations = measure_repeated(operation, warmup=1, repeats=repeats)
    return value, durations.to_dict()


def _finite_scalar(value: Array, /) -> float | None:
    number = float(np.asarray(value))
    return number if math.isfinite(number) else None


def _spectrum(size: int, rank: int, kind: Spectrum, /) -> HostFloat64[SingularDim]:
    kind = parse(kind, Spectrum, "spectrum")
    match kind:
        case "separated":
            leading = np.geomspace(8.0, 1.0, rank, dtype=np.float64)
            tail = 0.1 * np.power(0.9, np.arange(size - rank, dtype=np.float64))
            return np.concatenate((leading, tail))
        case "slow":
            return np.linspace(2.0, 1.0, size, dtype=np.float64)
        case "boundary":
            values = np.linspace(4.0, 0.2, size, dtype=np.float64)
            if rank < size:
                values[rank] = values[rank - 1]
            return values
        case "low-rank":
            return np.concatenate(
                (
                    np.geomspace(8.0, 1.0, rank, dtype=np.float64),
                    np.zeros(size - rank, dtype=np.float64),
                )
            )
        case _:
            raise ValueError("Unknown spectrum.")


def _host_frames(
    rows: int,
    columns: int,
    dtype: np.dtype,
    seed: int,
    /,
) -> tuple[HostInexact[TargetDim, SingularDim], HostInexact[SourceDim, SingularDim]]:
    generator = np.random.default_rng(seed)
    size = min(rows, columns)
    scope = Scope()
    left_values = generator.normal(size=(rows, size))
    right_values = generator.normal(size=(columns, size))
    if np.issubdtype(dtype, np.complexfloating):
        left = as_host_array(
            left_values + 1j * generator.normal(size=(rows, size)),
            HostInexact[TargetDim, SingularDim],
            "left_frame",
            scope=scope,
        )
        right = as_host_array(
            right_values + 1j * generator.normal(size=(columns, size)),
            HostInexact[SourceDim, SingularDim],
            "right_frame",
            scope=scope,
        )
    else:
        left = as_host_array(
            left_values, HostInexact[TargetDim, SingularDim], "left_frame", scope=scope
        )
        right = as_host_array(
            right_values, HostInexact[SourceDim, SingularDim], "right_frame", scope=scope
        )
    left_frame = np.linalg.qr(left, mode="reduced")[0]
    right_frame = np.linalg.qr(right, mode="reduced")[0]
    return (
        as_host_array(
            left_frame.astype(dtype, copy=False),
            HostInexact[TargetDim, SingularDim],
            "left_frame",
            scope=scope,
        ),
        as_host_array(
            right_frame.astype(dtype, copy=False),
            HostInexact[SourceDim, SingularDim],
            "right_frame",
            scope=scope,
        ),
    )


def _matrix_fixture(case: CampaignCase, /) -> Array:
    dtype = np.dtype(case.dtype)
    left, right = _host_frames(case.rows, case.columns, dtype, 701)
    spectrum = _spectrum(min(case.rows, case.columns), case.rank, case.spectrum)
    return jnp.asarray((left * spectrum[None, :]) @ right.conj().T, dtype=dtype)


def _policy(
    case: CampaignCase,
    /,
    *,
    differentiation: SVDDifferentiationMode = "none",
    require_leading: bool = False,
    resources: SVDResourcePolicy | None = None,
) -> SVDSolvePolicy:
    low_precision = np.dtype(case.dtype).itemsize <= (8 if "complex" in case.dtype else 4)
    tolerance = SVDTolerancePolicy(
        residual=2e-2 if case.randomized else (1e-5 if low_precision else 1e-9),
        orthogonality=1e-4 if low_precision else 1e-10,
    )
    method = (
        RandomizedSVD(
            oversampling=case.oversampling, power_iterations=case.power_iterations
        )
        if case.randomized
        else DenseSVD()
    )
    return SVDSolvePolicy(
        method,
        count=case.rank,
        tolerance=tolerance,
        approximation=SVDApproximationPolicy(require_leading=require_leading),
        differentiation=differentiation,
        resources=resources,
    )


def _dense_problem(
    matrix: Array,
    identifier: str,
    /,
    *,
    source_weight: Array | None = None,
    target_weight: Array | None = None,
) -> SVDProblem:
    dtype = np.dtype(matrix.dtype)
    source = la.ArraySpace(
        (matrix.shape[1],),
        dtype=dtype,
        pairing=None
        if source_weight is None
        else la.DiagonalPairing(source_weight, pairing_id=f"{identifier}:source-pairing"),
        space_id=f"{identifier}:source",
    )
    target = la.ArraySpace(
        (matrix.shape[0],),
        dtype=dtype,
        pairing=None
        if target_weight is None
        else la.DiagonalPairing(target_weight, pairing_id=f"{identifier}:target-pairing"),
        space_id=f"{identifier}:target",
    )
    return SVDProblem(
        la.DenseLinearOperator(
            matrix, source=source, target=target, operator_id=f"{identifier}:operator"
        ),
        problem_id=identifier,
    )


def _evidence(result: SVDSolveResult, /) -> dict[str, Any]:
    return {
        "status": int(np.asarray(result.status)),
        "primal_status": int(np.asarray(result.primal_status)),
        "derivative_status": int(np.asarray(result.derivative_status)),
        "leading_certified": bool(np.asarray(result.leading_evidence.certified)),
        "leading_gap_lower": _finite_scalar(result.leading_evidence.gap_lower_bound),
        "maximum_relative_residual": _finite_scalar(
            jnp.max(result.diagnostics.relative_residuals)
        ),
        "left_orthogonality_error": _finite_scalar(
            result.diagnostics.left_orthogonality_error
        ),
        "right_orthogonality_error": _finite_scalar(
            result.diagnostics.right_orthogonality_error
        ),
        "range_upper_bound": _finite_scalar(result.range_evidence.spectral_upper_bound),
        "range_failure_probability": _finite_scalar(
            result.range_evidence.failure_probability
        ),
        "rank_lower": int(np.asarray(result.rank_evidence.lower_bound)),
        "rank_upper": int(np.asarray(result.rank_evidence.upper_bound)),
        "rank_available": bool(np.asarray(result.rank_evidence.available)),
        "rank_deterministic_exact": bool(
            np.asarray(result.rank_evidence.deterministic_exact)
        ),
    }


def _stage_cost(plan: SVDSolvePlan, /) -> dict[str, Any]:
    cost = plan.cost
    return {
        "original_operator_bytes": cost.original_operator_bytes,
        "added_retained_bytes": cost.storage_bytes,
        "preparation_workspace_bytes": cost.preparation_workspace_bytes,
        "execution_workspace_bytes": cost.apply_workspace_bytes,
        "forward_column_actions": cost.operator_matvec_count,
        "adjoint_column_actions": cost.adjoint_matvec_count,
        "preparation_forward_block_calls": cost.preparation_forward_block_calls,
        "preparation_adjoint_block_calls": cost.preparation_adjoint_block_calls,
        "execution_forward_block_calls": cost.solve_forward_block_calls,
        "execution_adjoint_block_calls": cost.solve_adjoint_block_calls,
        "deterministic_scan_entries": cost.deterministic_scan_entries,
        "provider_workspace_exact": cost.provider_workspace_exact,
    }


def _native_case(case: CampaignCase, repeats: int, /) -> dict[str, Any]:
    matrix = _matrix_fixture(case)
    problem = _dense_problem(matrix, case.identifier)
    policy = _policy(case)
    key = jax.random.key(902) if case.randomized else None
    plan, planning = measure_host(lambda: plan_svd(problem, policy))
    _cold, preparation = measure_synchronized(lambda: prepare_svd(problem, plan, key=key))
    prepare_executable, prepare_compile = _compile_phase(
        lambda: _prepare_entry.lower(problem, plan, key)
    )
    prepared, warmed_preparation = _timed(
        lambda: prepare_executable(problem, plan, key), repeats
    )
    solve_executable, solve_compile = _compile_phase(lambda: _solve_entry.lower(prepared))
    result, first_execution = measure_synchronized(lambda: solve_executable(prepared))
    result, warmed_execution = _timed(lambda: solve_executable(prepared), repeats)
    updated = _dense_problem(
        matrix * jnp.asarray(1.01, dtype=matrix.real.dtype), case.identifier
    )
    refresh_executable, refresh_compile = _compile_phase(
        lambda: _refresh_entry.lower(prepared, updated)
    )
    refreshed, warmed_refresh = _timed(
        lambda: refresh_executable(prepared, updated), repeats
    )
    refresh_result = synchronize(_solve_entry(refreshed))
    source = problem.operator.source
    if not isinstance(source, la.ArraySpace):
        raise TypeError("Dense campaign source must be an ArraySpace.")
    probe = jnp.asarray(np.linspace(0.1, 1.0, case.columns), dtype=matrix.dtype)
    projection_executable, projection_compile = _compile_phase(
        lambda: _projection_entry.lower(result, source, probe)
    )
    projection, warmed_projection = _timed(
        lambda: projection_executable(result, source, probe), repeats
    )
    return {
        "case_id": case.identifier,
        "rows": case.rows,
        "columns": case.columns,
        "retained_rank": case.rank,
        "dtype": case.dtype,
        "spectrum": case.spectrum,
        "method": policy.method.name,
        "sketch_capacity": plan.sketch_size,
        "effective_oversampling": plan.effective_oversampling,
        "power_iterations": case.power_iterations if case.randomized else 0,
        "planning_seconds": planning,
        "cold_preparation_seconds": preparation,
        "prepare": {**prepare_compile, "warmed": warmed_preparation},
        "execute": {
            **solve_compile,
            "first_seconds": first_execution,
            "warmed": warmed_execution,
        },
        "refresh": {
            **refresh_compile,
            "warmed": warmed_refresh,
            "status": int(np.asarray(refresh_result.status)),
            "numeric_version": int(np.asarray(refreshed.numeric_version)),
        },
        "project": {
            **projection_compile,
            "warmed": warmed_projection,
            "norm": _finite_scalar(jnp.linalg.norm(projection)),
        },
        "logical_original_operator_bytes": logical_array_bytes(problem.operator),
        "logical_prepared_bytes": logical_array_bytes(prepared),
        "logical_state_bytes": logical_array_bytes(prepared.state),
        "logical_result_bytes": logical_array_bytes(result),
        "declared_cost": _stage_cost(plan),
        "last_step_evidence": _evidence(result),
    }


def _capacity_cases(capacities: tuple[int, ...], /) -> tuple[CampaignCase, ...]:
    shapes = sorted(
        {(size, 64) for size in capacities} | {(64, size) for size in capacities}
    )
    return tuple(
        CampaignCase(rows, columns, 4, randomized=route)
        for rows, columns in shapes
        for route in (False, True)
    )


def _grid_cases() -> tuple[CampaignCase, ...]:
    cases = [
        CampaignCase(
            128,
            64,
            rank,
            dtype=dtype,
            randomized=True,
            oversampling=oversampling,
            power_iterations=passes,
        )
        for rank in (4, 16)
        for dtype in ("float32", "float64", "complex64", "complex128")
        for oversampling in (0, 4, 8)
        for passes in (0, 1, 2)
    ]
    cases.extend(
        CampaignCase(128, 64, 4, spectrum=spectrum, randomized=True)
        for spectrum in ("slow", "boundary", "low-rank")
    )
    return tuple(cases)


def _projection_loss(
    problem: SVDProblem,
    policy: SVDSolvePolicy,
    key: PRNGKey | None,
    probe: Array,
    test: Array,
) -> Array:
    result = svd(problem, policy=policy, key=key)
    source = problem.operator.source
    if not isinstance(source, la.ArraySpace):
        raise TypeError("Qualification projector source must be an ArraySpace.")
    projected = projector_action(
        result.right_coordinates, result.right_response, source.riesz(probe)
    )
    return jnp.real(jnp.vdot(test, projected))


@partial(jax.jit, static_argnames=("identifier",))
def _projector_jvp_entry(
    matrix: Array,
    direction: Array,
    source_weight: Array | None,
    target_weight: Array | None,
    policy: SVDSolvePolicy,
    key: PRNGKey | None,
    identifier: str,
    probe: Array,
    test: Array,
) -> tuple[Array, Array]:
    def loss(value: Array) -> Array:
        problem = _dense_problem(
            value, identifier, source_weight=source_weight, target_weight=target_weight
        )
        return _projection_loss(problem, policy, key, probe, test)

    return jax.jvp(loss, (matrix,), (direction,))


def _derivative_observation(
    matrix: Array,
    direction: Array,
    policy: SVDSolvePolicy,
    key: PRNGKey | None,
    identifier: str,
    /,
    *,
    source_weight: Array | None = None,
    target_weight: Array | None = None,
    repeats: int = 3,
) -> dict[str, Any]:
    probe = jnp.asarray(np.linspace(0.2, 1.0, matrix.shape[1]), dtype=matrix.dtype)
    test = jnp.asarray(np.linspace(1.0, -0.4, matrix.shape[1]), dtype=matrix.dtype)

    def loss(value: Array) -> Array:
        problem = _dense_problem(
            value, identifier, source_weight=source_weight, target_weight=target_weight
        )
        return _projection_loss(problem, policy, key, probe, test)

    started = time.perf_counter()
    value, tangent = jax.jvp(loss, (matrix,), (direction,))
    synchronize((value, tangent))
    elapsed = time.perf_counter() - started
    epsilon = 1e-5
    reference = (
        loss(matrix + epsilon * direction) - loss(matrix - epsilon * direction)
    ) / (2 * epsilon)
    reverse = jnp.real(jnp.sum(jax.grad(loss)(matrix) * direction))
    synchronize((reference, reverse))
    executable, compilation = _compile_phase(
        lambda: _projector_jvp_entry.lower(
            matrix,
            direction,
            source_weight,
            target_weight,
            policy,
            key,
            identifier,
            probe,
            test,
        )
    )
    _, warmed = _timed(
        lambda: executable(
            matrix, direction, source_weight, target_weight, policy, key, probe, test
        ),
        repeats,
    )
    return {
        "jvp": float(np.asarray(tangent)),
        "finite_difference": float(np.asarray(reference)),
        "jvp_error": float(np.asarray(jnp.abs(tangent - reference))),
        "vjp_error": float(np.asarray(jnp.abs(reverse - tangent))),
        "cold_derivative_seconds": elapsed,
        "compiled_execution": {**compilation, "warmed": warmed},
    }


def _shell_problem() -> SVDProblem:
    left, right = _host_frames(128, 64, np.dtype(np.float64), 943)
    spectrum = np.asarray([8.0, 4.0, 2.0, 1.0], dtype=np.float64)
    source = la.ArraySpace((64,), dtype=np.float64, space_id="qualification-shell:source")
    target = la.ArraySpace(
        (128,), dtype=np.float64, space_id="qualification-shell:target"
    )
    operator = la.LowRankLinearOperator(
        jnp.asarray(left[:, :4] * spectrum[None, :]),
        jnp.asarray(right[:, :4]),
        source=source,
        target=target,
        operator_id="qualification-shell:factor-action",
    )
    return SVDProblem(
        NonmaterializingBenchmarkOperator(operator), problem_id="qualification-shell"
    )


def _resource_refusal(problem: SVDProblem, policy: SVDSolvePolicy, /) -> bool:
    limited = SVDSolvePolicy(
        policy.method,
        count=policy.count,
        tolerance=policy.tolerance,
        resources=SVDResourcePolicy(
            preparation_bytes=0, workspace_bytes=0, operator_matvecs=0
        ),
    )
    try:
        plan_svd(problem, limited)
    except ValueError:
        return True
    raise RuntimeError("Zero-budget SVD planning failed to refuse work.")


def _stopped_derivative_observation(
    problem: SVDProblem, policy: SVDSolvePolicy, key: PRNGKey, /
) -> dict[str, Any]:
    base = problem.operator
    if not isinstance(base, NonmaterializingBenchmarkOperator) or not isinstance(
        base.operator, la.LowRankLinearOperator
    ):
        raise TypeError("Qualification shell must own native rank factors.")
    low_rank = base.operator

    def stopped_loss(left: Array) -> Array:
        current = la.LowRankLinearOperator(
            left,
            low_rank.right_factor,
            source=low_rank.source,
            target=low_rank.target,
            operator_id=low_rank.operator_id,
        )
        fitted = svd(
            SVDProblem(
                NonmaterializingBenchmarkOperator(current), problem_id=problem.problem_id
            ),
            policy=policy,
            key=key,
        )
        return jnp.sum(fitted.singular_values)

    factor = low_rank.left_factor
    _, tangent = jax.jvp(stopped_loss, (factor,), (factor,))
    epsilon = 1e-5
    finite_difference = (
        stopped_loss((1 + epsilon) * factor) - stopped_loss((1 - epsilon) * factor)
    ) / (2 * epsilon)
    reverse = jnp.real(jnp.sum(jax.grad(stopped_loss)(factor) * factor))
    synchronize((tangent, finite_difference, reverse))
    return {
        "jvp": float(np.asarray(tangent)),
        "finite_difference": float(np.asarray(finite_difference)),
        "jvp_error": float(np.asarray(jnp.abs(tangent))),
        "vjp_error": float(np.asarray(jnp.abs(reverse))),
    }


def _qualification_fixture(
    profile: CapabilityProfile, repeats: int, /
) -> tuple[SVDProblem, SVDSolvePolicy, PRNGKey | None, dict[str, Any]]:
    attributes = dict(profile.support_tuples[0].attributes)
    raw_scenario = attributes["scenario"]
    if not isinstance(raw_scenario, str):
        raise TypeError("The qualification scenario must be a serialized identifier.")
    scenario = parse(
        SingularSubspaceScenario(raw_scenario), SingularSubspaceScenario, "scenario"
    )
    identifier = profile.profile_id
    limit = attributes["workspace_limit_bytes"]
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("The qualification workspace envelope must be an integer.")
    resources = SVDResourcePolicy(preparation_bytes=limit, workspace_bytes=limit)
    match scenario:
        case SingularSubspaceScenario.RANDOMIZED_SHELL:
            problem = _shell_problem()
            policy = _policy(
                CampaignCase(128, 64, 4, randomized=True, power_iterations=2),
                require_leading=True,
                resources=resources,
            )
            key = jax.random.key(617)
            return (
                problem,
                policy,
                key,
                _stopped_derivative_observation(problem, policy, key),
            )
        case SingularSubspaceScenario.DENSE_COMPLEX_DIAGONAL:
            left, right = _host_frames(8, 5, np.dtype(np.complex128), 117)
            source_weight = jnp.linspace(0.5, 2.0, 5, dtype=jnp.float64)
            target_weight = jnp.linspace(1.0, 3.0, 8, dtype=jnp.float64)
            reduced = (
                left * np.asarray([4.0, 4.0, 1.0, 0.5, 0.2])[None, :]
            ) @ right.conj().T
            matrix = jnp.asarray(reduced, dtype=jnp.complex128)
            matrix = (
                matrix
                * jnp.sqrt(source_weight).astype(matrix.dtype)[None, :]
                / jnp.sqrt(target_weight).astype(matrix.dtype)[:, None]
            )
            problem = _dense_problem(
                matrix,
                identifier,
                source_weight=source_weight,
                target_weight=target_weight,
            )
            policy = _policy(
                CampaignCase(8, 5, 2, dtype="complex128"),
                differentiation="projector",
                resources=resources,
            )
            direction = jnp.asarray(
                np.random.default_rng(719).normal(size=(8, 5))
                + 1j * np.random.default_rng(720).normal(size=(8, 5)),
                dtype=jnp.complex128,
            )
            derivative = _derivative_observation(
                matrix,
                direction,
                policy,
                None,
                identifier,
                source_weight=source_weight,
                target_weight=target_weight,
                repeats=repeats,
            )
            return problem, policy, None, derivative
        case SingularSubspaceScenario.RANDOMIZED_RESIDENT:
            case = CampaignCase(
                64, 32, 4, randomized=True, oversampling=4, power_iterations=1
            )
            matrix = _matrix_fixture(case)
            problem = _dense_problem(matrix, identifier)
            policy = _policy(
                case,
                differentiation="projector",
                require_leading=True,
                resources=resources,
            )
            key = jax.random.key(618)
            direction = jnp.asarray(
                np.random.default_rng(721).normal(size=matrix.shape), dtype=matrix.dtype
            )
            return (
                problem,
                policy,
                key,
                _derivative_observation(
                    matrix, direction, policy, key, identifier, repeats=repeats
                ),
            )
        case SingularSubspaceScenario.DENSE_REPEATED:
            diagonal = jnp.diag(jnp.asarray([3.0, 3.0, 1.0], dtype=jnp.float64))
            matrix = jnp.concatenate((diagonal, -diagonal), axis=0)
            problem = _dense_problem(matrix, identifier)
            policy = _policy(
                CampaignCase(6, 3, 2), differentiation="projector", resources=resources
            )
            direction = jnp.asarray(
                np.random.default_rng(722).normal(size=matrix.shape), dtype=matrix.dtype
            )
            return (
                problem,
                policy,
                None,
                _derivative_observation(
                    matrix, direction, policy, None, identifier, repeats=repeats
                ),
            )
        case _:
            raise ValueError("Unknown singular-subspace qualification scenario.")


def _qualification(repeats: int, /) -> dict[str, Any]:
    observations = []
    rows = []
    for profile in singular_subspace_candidate_profiles():
        problem, policy, key, derivative = _qualification_fixture(profile, repeats)
        identifier = profile.profile_id
        plan = plan_svd(problem, policy)
        prepared = synchronize(prepare_svd(problem, plan, key=key))
        result, execution = _timed(
            lambda prepared=prepared: _solve_entry(prepared), repeats
        )
        evidence = _evidence(result)
        residual = evidence["maximum_relative_residual"]
        left_error = evidence["left_orthogonality_error"]
        right_error = evidence["right_orthogonality_error"]
        gates = {
            "original-triplet-residual": residual is not None
            and residual <= policy.tolerance.residual,
            "metric-orthogonality": left_error is not None
            and right_error is not None
            and max(left_error, right_error) <= policy.tolerance.orthogonality,
            "rank-evidence": bool(np.asarray(result.rank_evidence.available)),
            "leading-selection": bool(np.asarray(result.leading_evidence.certified)),
            "requested-derivative": (
                policy.differentiation == "none"
                and not bool(np.asarray(result.derivative_valid))
                and abs(derivative["finite_difference"]) > 1e-4
                or bool(np.asarray(result.derivative_valid))
            )
            and derivative["jvp_error"] < 2e-5
            and derivative["vjp_error"] < 2e-5,
            "resource-refusal": _resource_refusal(problem, policy),
            "stage-admission": plan.cost.accepted,
        }
        metrics = {
            "native_status": float(evidence["status"]),
            "logical_state_bytes": float(logical_array_bytes(prepared.state)),
            "jvp": derivative["jvp"],
            "finite_difference": derivative["finite_difference"],
            "jvp_error": derivative["jvp_error"],
            "vjp_error": derivative["vjp_error"],
        }
        observation = CoreQualificationObservation(profile, gates, metrics)
        observations.append(observation)
        rows.append(
            {
                "profile_id": identifier,
                "evidence": evidence,
                "derivative": derivative,
                "warmed_execution": execution,
                "declared_cost": _stage_cost(plan),
            }
        )
    return {"record": singular_subspace_qualification_record(observations), "cases": rows}


def _ml_smoke() -> dict[str, Any]:
    diagonal = jnp.diag(jnp.asarray([3.0, 3.0, 1.0], dtype=jnp.float64))
    values = jnp.concatenate((diagonal, -diagonal), axis=0)
    weights = jnp.ones((6,), dtype=jnp.float64)
    probe = jnp.asarray([[0.2, -0.4, 0.7], [0.8, 0.3, -0.2]], dtype=jnp.float64)
    recipe = PCA(2, differentiate="projector")

    def projection(value: Array) -> Array:
        fit = recipe.fit_batch(MLBatch(value, sample_weight=weights))
        model = fit.as_trainable()
        if not isinstance(model, SubspaceModel):
            raise TypeError("PCA smoke did not produce a SubspaceModel.")
        return jnp.sum(
            model.project(probe)
            * jnp.asarray([[0.5, 0.1, -0.3], [-0.4, 0.8, 0.2]], dtype=jnp.float64)
        )

    direction = jnp.asarray(
        np.random.default_rng(41).normal(size=values.shape), dtype=jnp.float64
    )
    _, tangent = jax.jvp(projection, (values,), (direction,))
    epsilon = 1e-5
    reference = (
        projection(values + epsilon * direction)
        - projection(values - epsilon * direction)
    ) / (2 * epsilon)
    pod = POD(
        2, physical_weights=jnp.asarray([0.5, 1.5, 2.0], dtype=jnp.float64)
    ).fit_batch(MLBatch(values))
    pod_model = pod.as_trainable()
    if not isinstance(pod_model, SubspaceModel):
        raise TypeError("Physical POD smoke did not produce a SubspaceModel.")
    reconstructed = pod_model.inverse_transform(pod_model.transform(probe))
    projected = pod_model.project(probe)
    synchronize((tangent, reference, reconstructed, projected))
    return {
        "repeated_projector_jvp": float(np.asarray(tangent)),
        "finite_difference": float(np.asarray(reference)),
        "jvp_error": float(np.asarray(jnp.abs(tangent - reference))),
        "physical_projection_defect": float(
            np.asarray(jnp.max(jnp.abs(reconstructed - projected)))
        ),
        "physical_fit_valid": bool(np.asarray(pod.valid)),
        "logical_fitted_model_bytes": logical_array_bytes(pod_model),
    }


def _operator_smoke() -> dict[str, Any]:
    sensor = OperatorAxis("sensor", jnp.asarray([0.0], dtype=jnp.float64))
    query = OperatorAxis(
        "x",
        jnp.asarray([0.0, 0.2, 0.7, 1.0], dtype=jnp.float64),
        quadrature_weights=jnp.asarray([0.1, 0.3, 0.4, 0.2], dtype=jnp.float64),
    )
    coefficients = jnp.asarray(
        [[-2.0, 0.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 1.0], [2.0, -1.0]],
        dtype=jnp.float64,
    )
    modes = jnp.asarray([[1.0, 0.5, -0.2, 0.8], [0.0, 1.0, 0.4, -0.5]], dtype=jnp.float64)
    values = coefficients @ modes + jnp.asarray([3.0, -2.0, 1.0, 0.5], dtype=jnp.float64)
    dataset = operator_dataset_from_arrays(
        {"source": jnp.linspace(-1.0, 1.0, 5, dtype=jnp.float64)[:, None]},
        {"state": values},
        source_axes={"source": (sensor,)},
        query_axes=(query,),
    )
    fitted = fit_operator_pod(
        dataset, "state", 2, centered=True, require_physical_quadrature=True
    )
    encoding = fitted.transform(values)
    decoding = fitted.inverse_transform(encoding)
    projected = fitted.project(values)
    synchronize((encoding, decoding, projected))
    return {
        "valid": bool(np.asarray(fitted.valid)),
        "reconstruction_error": float(np.asarray(jnp.max(jnp.abs(decoding - values)))),
        "projection_error": float(np.asarray(jnp.max(jnp.abs(projected - decoding)))),
        "logical_fitted_model_bytes": logical_array_bytes(fitted),
    }


def _incremental_smoke() -> dict[str, float]:
    first = jnp.asarray(
        [[3.0, 0.1, 0.4], [-3.0, -0.1, -0.4], [0.2, 2.0, 0.3], [-0.2, -2.0, -0.3]],
        dtype=jnp.float64,
    )
    second = jnp.asarray(
        [[2.0, -0.2, 0.5], [-2.0, 0.2, -0.5], [0.1, 1.5, -0.2], [-0.1, -1.5, 0.2]],
        dtype=jnp.float64,
    )
    probe = jnp.asarray([[0.4, -0.1, 0.8]], dtype=jnp.float64)
    recipe = IncrementalPCA(2)

    def loss(values: Array) -> Array:
        initial = recipe.fit_batch(MLBatch(values)).as_trainable()
        if not isinstance(initial, IncrementalPCAModel):
            raise TypeError("Incremental smoke did not produce an IncrementalPCAModel.")
        model = initial.update_recipe().fit_batch(MLBatch(second)).as_trainable()
        if not isinstance(model, IncrementalPCAModel):
            raise TypeError(
                "Incremental merge smoke did not produce an IncrementalPCAModel."
            )
        return jnp.sum(
            model.project(probe) * jnp.asarray([[0.2, 0.3, -0.4]], dtype=jnp.float64)
        )

    direction = jnp.asarray(
        np.random.default_rng(57).normal(size=first.shape), dtype=jnp.float64
    )
    _, tangent = jax.jvp(loss, (first,), (direction,))
    epsilon = 1e-5
    reference = (
        loss(first + epsilon * direction) - loss(first - epsilon * direction)
    ) / (2 * epsilon)
    synchronize((tangent, reference))
    return {
        "earlier_chunk_jvp": float(np.asarray(tangent)),
        "finite_difference": float(np.asarray(reference)),
        "jvp_error": float(np.asarray(jnp.abs(tangent - reference))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacities", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/singular_subspaces.json")
    )
    arguments = parser.parse_args()
    capacities = tuple(sorted(set(arguments.capacities)))
    if not capacities or any(size < 16 or size > 1024 for size in capacities):
        raise ValueError("Capacities must lie between 16 and 1024.")
    if arguments.repeats < 1:
        raise ValueError("Repeats must be positive.")
    jax.config.update("jax_enable_x64", True)
    cases = _capacity_cases(capacities) + (_grid_cases() if arguments.grid else ())
    ordered = tuple(
        sorted(
            {case.identifier: case for case in cases}.values(),
            key=lambda case: case.identifier,
        )
    )
    records = []
    for case in ordered:
        row = _native_case(case, arguments.repeats)
        records.append(row)
        print(
            json.dumps(
                {"case_id": case.identifier, **row["last_step_evidence"]},
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
    smoke = {
        "ml": _ml_smoke(),
        "operator_pod": _operator_smoke(),
        "incremental": _incremental_smoke(),
    }
    qualification = _qualification(arguments.repeats)
    if (
        any(value["jvp_error"] >= 2e-5 for value in (smoke["ml"], smoke["incremental"]))
        or not smoke["operator_pod"]["valid"]
        or not qualification["record"]["passed"]
    ):
        raise RuntimeError("Singular-subspace runtime smoke or qualification failed.")
    root = Path(__file__).resolve().parents[1]
    identity = capture_benchmark_identity(
        root, Path(__file__), records[0]["last_step_evidence"].keys()
    )
    record = {
        "identity": identity.to_dict(),
        "environment": capture_environment().to_dict(),
        "cases": records,
        "smoke": smoke,
        "qualification": qualification,
        "memory_observation": "Compiler estimates and logical retained arrays; no measured peak-device-memory claim.",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "rows": len(records),
                "smoke": smoke,
                "qualification_passed": qualification["record"]["passed"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
