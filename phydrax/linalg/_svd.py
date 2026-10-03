# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import NamedTuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from .._differentiation import DerivativeRoute
from ..typing import PRNGKey
from ._operators import DenseLinearOperator, estimate_operator_action_cost
from ._singular_subspaces import (
    attach_selected_triplet_derivative,
    canonicalize_singular_triplets,
    row_phase_evidence,
    selected_singular_responses,
    singular_value_response,
    SingularSubspaceResponse,
)
from ._spaces import _coordinate_dtype, _has_diagonal_pairing, AbstractVectorSpace
from ._svd_contracts import (
    DenseSVD,
    DenseSVDState,
    PreparedSVDSolve,
    RandomizedSVD,
    RandomizedSVDState,
    SVDCertificateKind,
    SVDCostEstimate,
    SVDLeadingEvidence,
    SVDProblem,
    SVDRangeEvidence,
    SVDSolveDiagnostics,
    SVDSolvePlan,
    SVDSolvePolicy,
    SVDSolveProvenance,
    SVDSolveResult,
    SVDSolveStatus,
)
from ._svd_dense import prepare_dense, restore_coordinates, thin_decomposition
from ._svd_evidence import (
    compressed_spectrum_allowance,
    discarded_singular_maximum,
    leading_evidence,
    rank_evidence,
    spectral_margins,
    triplet_evidence,
)
from ._svd_randomized import prepare_randomized


_T = TypeVar("_T")


def _stop_arrays(value: _T, /) -> _T:
    return jax.tree.map(
        lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_array(leaf) else leaf, value
    )


class _SVDCostPhases(NamedTuple):
    retained: int
    preparation: int
    workspace: int
    forward: int
    backward: int
    forward_calls: int
    backward_calls: int
    scan: int
    scratch_width: int


def _randomized_cost_phases(
    problem: SVDProblem,
    policy: SVDSolvePolicy,
    method: RandomizedSVD,
    width: int,
    resident: bool,
    itemsize: int,
    scale_entries: int,
    /,
) -> _SVDCostPhases:
    rows, columns, count = (
        problem.operator.target.size,
        problem.operator.source.size,
        policy.count,
    )
    q = method.power_iterations
    audit = 0 if resident else policy.approximation.audit_probes
    retained = itemsize * (scale_entries + (rows + columns) * width) + 128
    preparation = retained + itemsize * (
        (rows + columns) * max(width, audit, min(32, columns) if resident else 0)
        + 4 * width * width
    )
    workspace = retained + itemsize * (
        5 * width * width + (rows + columns) * (width + 4 * count) + 4 * count * count
    )
    if policy.differentiation != "none":
        derivative_tape = (
            itemsize * (q + 1) * (4 * (rows + columns) * width + 8 * width * width)
        )
        preparation += derivative_tape
        workspace += derivative_tape
    forward, backward = (q + 1) * width + audit, (q + 1) * width
    forward_calls, backward_calls = q + 1 + (audit > 0), q + 1
    scan = rows * columns if resident else 0
    scratch_width = max(
        width, policy.approximation.audit_probes if not resident else 0, count
    )
    return _SVDCostPhases(
        retained,
        preparation,
        workspace,
        forward,
        backward,
        forward_calls,
        backward_calls,
        scan,
        scratch_width,
    )


def _dense_cost_phases(
    rows: int, columns: int, count: int, itemsize: int, scale_entries: int, /
) -> _SVDCostPhases:
    retained = itemsize * (rows * columns + scale_entries) + 8
    preparation, workspace = (
        4 * retained,
        3 * retained + itemsize * ((rows + columns) * 4 * count + 4 * count * count),
    )
    return _SVDCostPhases(retained, preparation, workspace, columns, 0, 0, 0, 0, columns)


def _cost_refusals(
    problem: SVDProblem,
    policy: SVDSolvePolicy,
    preparation: int,
    workspace: int,
    operator_matvecs: int,
    itemsize: int,
    diagonal: bool,
    /,
) -> list[str]:
    checks = (
        (preparation, policy.resources.preparation_bytes, "preparation bytes"),
        (workspace, policy.resources.workspace_bytes, "workspace bytes"),
        (operator_matvecs, policy.resources.operator_matvecs, "operator matvecs"),
    )
    failures = [
        f"{name} estimate {required} exceeds budget {budget}"
        for required, budget, name in checks
        if required > budget
    ]
    if isinstance(policy.method, DenseSVD):
        operator = problem.operator
        rows, columns = operator.target.size, operator.source.size
        if rows * columns > policy.materialization.max_entries:
            failures.append("dense entries exceed materialization limit")
        if rows * columns * itemsize > policy.materialization.max_bytes:
            failures.append("dense bytes exceed materialization limit")
        if not operator.capabilities.materialize:
            failures.append("operator does not support materialization")
        if (
            not diagonal
            and max(rows * rows, columns * columns) > policy.materialization.max_entries
        ):
            failures.append("pairing entries exceed materialization limit")
        if (
            not diagonal
            and max(rows * rows, columns * columns) * itemsize
            > policy.materialization.max_bytes
        ):
            failures.append("pairing bytes exceed materialization limit")
    return failures


def _cost(
    problem: SVDProblem, policy: SVDSolvePolicy, width: int, resident: bool, /
) -> SVDCostEstimate:
    operator = problem.operator
    rows, columns, count = operator.target.size, operator.source.size, policy.count
    itemsize = _coordinate_dtype(operator.source).itemsize
    provider = estimate_operator_action_cost(operator)
    diagonal = _has_diagonal_pairing(operator.source) and _has_diagonal_pairing(
        operator.target
    )
    scale_entries = rows + columns if diagonal else rows * rows + columns * columns
    method = policy.method
    if isinstance(method, RandomizedSVD):
        phases = _randomized_cost_phases(
            problem, policy, method, width, resident, itemsize, scale_entries
        )
    elif isinstance(method, DenseSVD):
        phases = _dense_cost_phases(rows, columns, count, itemsize, scale_entries)
    else:
        raise TypeError("Unsupported SVD method.")
    preparation = (
        phases.preparation + provider.apply_workspace_bytes_per_rhs * phases.scratch_width
    )
    workspace = phases.workspace + provider.apply_workspace_bytes_per_rhs * max(
        count, width
    )
    failures = _cost_refusals(
        problem,
        policy,
        preparation,
        workspace,
        phases.forward + phases.backward + 2 * count,
        itemsize,
        diagonal,
    )
    fused = operator.supports_fused_block_action
    return SVDCostEstimate(
        phases.retained,
        provider.storage_bytes,
        preparation,
        workspace,
        phases.forward,
        phases.backward,
        phases.forward_calls if fused else phases.forward,
        phases.backward_calls if fused else phases.backward,
        1 if fused else count,
        1 if fused else count,
        phases.scan,
        provider.exact,
        not failures,
        "; ".join(failures) if failures else "method fits declared stage budgets",
    )


def _svd_method_geometry(
    problem: SVDProblem, selected: SVDSolvePolicy, resident: bool, /
) -> tuple[int, SVDCertificateKind]:
    if isinstance(selected.method, RandomizedSVD):
        if selected.which != "largest":
            raise ValueError("RandomizedSVD supports which='largest' only.")
        if not (
            _has_diagonal_pairing(problem.operator.source)
            and _has_diagonal_pairing(problem.operator.target)
        ):
            raise TypeError(
                "RandomizedSVD requires admitted coordinate-diagonal pairings."
            )
        width = min(selected.count + selected.method.oversampling, problem.maximum_rank)
        kind = "deterministic-frobenius" if resident else "independent-gaussian"
        if selected.rank.require_full_rank and not (
            resident and width == problem.maximum_rank
        ):
            raise ValueError(
                "Randomized full-rank policy requires full coverage and deterministic global evidence."
            )
    elif isinstance(selected.method, DenseSVD):
        width, kind = problem.maximum_rank, "exact-spectrum"
    else:
        raise TypeError("Unsupported SVD method.")
    return width, kind


def plan_svd(
    problem: SVDProblem, policy: SVDSolvePolicy | None = None, /
) -> SVDSolvePlan:
    if not isinstance(problem, SVDProblem):
        raise TypeError("problem must be an SVDProblem.")
    selected = SVDSolvePolicy() if policy is None else policy
    if not isinstance(selected, SVDSolvePolicy):
        raise TypeError("policy must be an SVDSolvePolicy.")
    if selected.count > problem.maximum_rank:
        raise ValueError("Requested SVD count exceeds min(target size, source size).")
    if (
        selected.approximation.require_leading
        and selected.which == "smallest"
        and selected.count < problem.maximum_rank
    ):
        raise ValueError(
            "A partial smallest-mode selection cannot require a leading subspace."
        )
    resident = isinstance(problem.operator, DenseLinearOperator)
    width, kind = _svd_method_geometry(problem, selected, resident)
    return SVDSolvePlan(
        problem, selected, _cost(problem, selected, width, resident), width, kind
    )


def _prepare_state(
    problem: SVDProblem, plan: SVDSolvePlan, key: PRNGKey | None, version: Array, /
) -> DenseSVDState | RandomizedSVDState:
    if isinstance(plan.policy.method, DenseSVD):
        if key is not None:
            raise ValueError("DenseSVD does not accept a randomized key.")
        return prepare_dense(problem, plan)
    if isinstance(plan.policy.method, RandomizedSVD):
        if key is None:
            raise ValueError("RandomizedSVD requires an explicit scalar typed key.")
        return prepare_randomized(problem, plan, key, version)
    raise TypeError("Unsupported SVD method.")


def _checked_preparation(
    state: DenseSVDState | RandomizedSVDState, plan: SVDSolvePlan, /
) -> DenseSVDState | RandomizedSVDState:
    if plan.policy.failure.mode == "status":
        return state
    failed = state.preparation_status != int(SVDSolveStatus.SUCCESS)
    message = "SVD numerical preparation failed; inspect status-mode evidence."
    if isinstance(state, DenseSVDState):
        return eqx.tree_at(
            lambda value: (value.preparation_status, value.reduced_operator),
            state,
            (
                eqx.error_if(state.preparation_status, failed, message),
                eqx.error_if(state.reduced_operator, failed, message),
            ),
        )
    return eqx.tree_at(
        lambda value: (
            value.preparation_status,
            value.range_basis,
            value.compressed_operator,
        ),
        state,
        (
            eqx.error_if(state.preparation_status, failed, message),
            eqx.error_if(state.range_basis, failed, message),
            eqx.error_if(state.compressed_operator, failed, message),
        ),
    )


def prepare_svd(
    problem: SVDProblem,
    policy: SVDSolvePolicy | SVDSolvePlan | None = None,
    /,
    *,
    key: PRNGKey | None = None,
) -> PreparedSVDSolve:
    if isinstance(policy, SVDSolvePlan):
        plan = policy
        current = plan_svd(problem, plan.policy)
        if plan.plan_id != current.plan_id:
            raise ValueError("SVD plan does not match the problem structure.")
    else:
        plan = plan_svd(problem, policy)
    version = jnp.asarray(0, jnp.int32)
    state = _prepare_state(problem, plan, key, version)
    state = _checked_preparation(state, plan)
    return PreparedSVDSolve(problem, plan, state, version)


def refresh_svd(prepared: PreparedSVDSolve, problem: SVDProblem, /) -> PreparedSVDSolve:
    if not isinstance(prepared, PreparedSVDSolve) or not isinstance(problem, SVDProblem):
        raise TypeError("Expected PreparedSVDSolve and SVDProblem.")
    plan = plan_svd(problem, prepared.plan.policy)
    if prepared.plan.plan_id != plan.plan_id:
        raise ValueError("SVD refresh changed symbolic structure.")
    old_version = eqx.error_if(
        prepared.numeric_version,
        prepared.numeric_version == jnp.iinfo(jnp.int32).max,
        "SVD numerical version overflow.",
    )
    version = old_version + jnp.asarray(1, jnp.int32)
    key = (
        prepared.state.root_key
        if isinstance(prepared.state, RandomizedSVDState)
        else None
    )
    state = _prepare_state(problem, plan, key, version)
    state = _checked_preparation(state, plan)
    return PreparedSVDSolve(problem, plan, state, version)


class _Factors(NamedTuple):
    matrix: Array
    left: Array
    values: Array
    right: Array
    indices: Array
    eta: Array
    allowance: Array
    probability: Array
    audit_maximum: Array
    total_energy: Array | None
    qr_margin: Array
    qr_valid: Array


def _factors(prepared: PreparedSVDSolve, /) -> _Factors:
    state, plan = prepared.state, prepared.plan
    available = state.preparation_status == int(SVDSolveStatus.SUCCESS)
    if isinstance(state, DenseSVDState):
        matrix = state.reduced_operator
        zero = jnp.asarray(0, matrix.real.dtype)
        eta, allowance, probability, maximum = (
            zero,
            zero,
            zero,
            jnp.asarray(jnp.nan, matrix.real.dtype),
        )
        energy, margin, qr_valid = (
            jnp.sum(jnp.abs(matrix) ** 2),
            jnp.asarray(jnp.inf, matrix.real.dtype),
            jnp.asarray(True),
        )
    else:
        matrix = state.compressed_operator
        eta, allowance, probability, maximum = (
            state.range_bound,
            state.range_allowance,
            state.audit_failure_probability,
            state.audit_maximum_norm,
        )
        energy, margin, qr_valid = (
            state.total_energy,
            state.qr_minimum_margin,
            state.qr_full_rank,
        )
    method = plan.policy.method
    algorithm = method.algorithm if isinstance(method, DenseSVD) else "divide-and-conquer"
    left, values, right = _stop_arrays(thin_decomposition(matrix, available, algorithm))
    left, values, right = canonicalize_singular_triplets(left, values, right)
    order = (
        jnp.arange(values.shape[0])
        if plan.policy.which == "largest"
        else jnp.arange(values.shape[0] - 1, -1, -1)
    )
    return _Factors(
        matrix,
        left,
        values,
        right,
        order[: plan.policy.count],
        eta,
        allowance,
        probability,
        maximum,
        energy,
        margin,
        qr_valid,
    )


def _restore(
    prepared: PreparedSVDSolve, left: Array, right: Array, /
) -> tuple[Array, Array]:
    state = prepared.state
    if isinstance(state, DenseSVDState):
        return restore_coordinates(
            state.target_factor, left, state.diagonal
        ), restore_coordinates(state.source_factor, right, state.diagonal)
    return (state.range_basis @ left) / state.target_factor[
        :, None
    ], right / state.source_factor[:, None]


def _attached_responses(
    prepared: PreparedSVDSolve, factors: _Factors, valid: Array, /
) -> tuple[Array, Array, Array, SingularSubspaceResponse, SingularSubspaceResponse]:
    mode = prepared.plan.policy.differentiation
    selected_left, selected_right, selected_values = (
        factors.left[:, factors.indices],
        factors.right[:, factors.indices],
        factors.values[factors.indices],
    )
    left_response = jnp.zeros(
        (prepared.problem.operator.target.size, selected_values.shape[0]),
        factors.matrix.dtype,
    )
    right_response = jnp.zeros(
        (prepared.problem.operator.source.size, selected_values.shape[0]),
        factors.matrix.dtype,
    )
    left_core = right_core = jnp.zeros(
        (selected_values.shape[0], selected_values.shape[0]), factors.matrix.dtype
    )
    arguments = (
        factors.matrix,
        factors.left,
        factors.values,
        factors.right,
        factors.indices,
        valid,
    )
    if mode == "basis":
        selected_left, selected_values, selected_right = (
            attach_selected_triplet_derivative(*arguments)
        )
        left, right = _restore(prepared, selected_left, selected_right)
        left, selected_values, right = canonicalize_singular_triplets(
            left, selected_values, right
        )
    elif mode == "projector":
        frame_left, frame_right, core_left, core_right = selected_singular_responses(
            *arguments
        )
        state = prepared.state
        if isinstance(state, RandomizedSVDState):
            lifted = state.range_basis @ frame_left
            reference = jax.lax.stop_gradient(state.range_basis @ selected_left)
            correction = lifted - jax.lax.stop_gradient(lifted)
            connection = reference.conj().T @ correction
            horizontal = lifted - reference @ connection
            diagonal_core = jnp.diag(selected_values**2).astype(factors.matrix.dtype)
            core_left = (
                core_left
                + connection @ diagonal_core
                + diagonal_core @ connection.conj().T
            )
            restored_left = horizontal / state.target_factor[:, None]
            restored_right = frame_right / state.source_factor[:, None]
        else:
            restored_left, restored_right = _restore(prepared, frame_left, frame_right)
        raw_left, raw_right = _stop_arrays(
            _restore(prepared, selected_left, selected_right)
        )
        left, _, right = canonicalize_singular_triplets(
            raw_left, selected_values, raw_right
        )
        phases = jnp.sum(raw_right.conj() * right, axis=0) / jnp.sum(
            jnp.abs(raw_right) ** 2, axis=0
        ).astype(raw_right.dtype)
        phases = jax.lax.stop_gradient(phases)
        restored_left, restored_right = (
            restored_left * phases[None, :],
            restored_right * phases[None, :],
        )
        core_left = phases.conj()[:, None] * core_left * phases[None, :]
        core_right = phases.conj()[:, None] * core_right * phases[None, :]
        left_response = restored_left - jax.lax.stop_gradient(restored_left)
        right_response = restored_right - jax.lax.stop_gradient(restored_right)
        left_core, right_core = (
            core_left - jax.lax.stop_gradient(core_left),
            core_right - jax.lax.stop_gradient(core_right),
        )
    else:
        left, right = _stop_arrays(_restore(prepared, selected_left, selected_right))
        left, selected_values, right = canonicalize_singular_triplets(
            left, selected_values, right
        )
        if mode == "singular-values":
            selected_values = singular_value_response(*arguments)
    return (
        left,
        selected_values,
        right,
        SingularSubspaceResponse(left_response, left_core),
        SingularSubspaceResponse(right_response, right_core),
    )


def _responses(
    prepared: PreparedSVDSolve, factors: _Factors, valid: Array, /
) -> tuple[Array, Array, Array, SingularSubspaceResponse, SingularSubspaceResponse]:
    count, dtype = prepared.plan.policy.count, factors.matrix.dtype

    def unavailable(
        _: Array,
    ) -> tuple[Array, Array, Array, SingularSubspaceResponse, SingularSubspaceResponse]:
        left = jnp.full((prepared.problem.operator.target.size, count), jnp.nan, dtype)
        right = jnp.full((prepared.problem.operator.source.size, count), jnp.nan, dtype)
        values = jnp.full((count,), jnp.nan, factors.values.dtype)
        core = jnp.zeros((count, count), dtype)
        return (
            left,
            values,
            right,
            SingularSubspaceResponse(jnp.zeros_like(left), core),
            SingularSubspaceResponse(jnp.zeros_like(right), core),
        )

    available = (prepared.state.preparation_status == 0) & jnp.all(
        jnp.isfinite(factors.values)
    )
    return jax.lax.cond(
        available,
        lambda _: _attached_responses(prepared, factors, valid),
        unavailable,
        jnp.asarray(0),
    )


class _Admission(NamedTuple):
    primal_status: Array
    valid: Array
    derivative_status: Array
    aggregate_status: Array
    isolation: Array
    boundary: Array
    pivot_magnitudes: Array
    pivot_gaps: Array
    leading: SVDLeadingEvidence


def _admission(
    prepared: PreparedSVDSolve,
    factors: _Factors,
    relative: Array,
    orthogonality: Array,
    /,
) -> _Admission:
    policy = prepared.plan.policy
    finite = jnp.all(jnp.isfinite(factors.values)) & jnp.all(jnp.isfinite(relative))
    converged = jnp.all(relative <= policy.tolerance.residual) & (
        orthogonality <= policy.tolerance.orthogonality
    )
    status = jnp.where(
        finite,
        jnp.where(
            converged,
            int(SVDSolveStatus.SUCCESS),
            int(SVDSolveStatus.RESIDUAL_TOLERANCE_NOT_MET),
        ),
        int(SVDSolveStatus.NONFINITE_OUTPUT),
    ).astype(jnp.int32)
    available = prepared.state.preparation_status == int(SVDSolveStatus.SUCCESS)
    status = jnp.where(available, status, prepared.state.preparation_status)
    rank = rank_evidence(
        factors.values,
        factors.eta,
        finite & available,
        prepared.problem,
        prepared.plan,
        factors.probability,
    )
    if policy.rank.require_full_rank:
        rank_status = jnp.where(
            rank.available
            & rank.deterministic_exact
            & (rank.lower_bound == rank.upper_bound),
            jnp.where(
                rank.lower_bound == prepared.problem.maximum_rank,
                status,
                int(SVDSolveStatus.RANK_DEFICIENT),
            ),
            int(SVDSolveStatus.GLOBAL_RANK_UNCERTIFIED),
        )
        status = jnp.where(available & finite, rank_status, status)
    spectrum_allowance = compressed_spectrum_allowance(
        factors.values, prepared.problem, prepared.plan
    )
    leading = leading_evidence(
        factors.values,
        factors.eta,
        policy.count,
        prepared.plan.sketch_size == prepared.problem.maximum_rank,
        available & finite,
        factors.allowance + 2 * spectrum_allowance,
        policy.which,
    )
    if policy.approximation.require_leading:
        status = jnp.where(
            (status == 0) & ~leading.certified,
            int(SVDSolveStatus.LEADING_UNCERTIFIED),
            status,
        )
    isolation, boundary = spectral_margins(
        factors.values, factors.indices, factors.matrix.shape[0], factors.matrix.shape[1]
    )
    _, physical_right = _stop_arrays(
        _restore(
            prepared, factors.left[:, factors.indices], factors.right[:, factors.indices]
        )
    )
    pivots, pivot_gaps = row_phase_evidence(physical_right.T)
    selected = factors.values[factors.indices]
    positive = jnp.all(selected > rank.threshold_upper)
    mode = policy.differentiation
    if mode == "none":
        valid = jnp.asarray(False)
        derivative_status = jnp.asarray(int(SVDSolveStatus.SUCCESS), jnp.int32)
    else:
        full_identity = policy.count == factors.matrix.shape[0] == factors.matrix.shape[1]
        valid = (status == 0) & factors.qr_valid
        if mode == "projector":
            valid = valid & ((positive & (boundary > 0)) | full_identity)
        else:
            valid = valid & positive & jnp.all(isolation > 0)
        if mode == "basis":
            pivot_allowance = 64 * jnp.finfo(selected.dtype).eps
            valid = (
                valid
                & jnp.all(pivots > pivot_allowance)
                & jnp.all(pivot_gaps > pivot_allowance)
            )
        derivative_status = jnp.where(
            valid, 0, int(SVDSolveStatus.DIFFERENTIATION_REJECTED)
        ).astype(jnp.int32)
    aggregate = jnp.where(
        (status == 0) & (derivative_status != 0), derivative_status, status
    )
    return _Admission(
        status,
        valid,
        derivative_status,
        aggregate,
        isolation,
        boundary,
        pivots,
        pivot_gaps,
        leading,
    )


def _assemble(prepared: PreparedSVDSolve, factors: _Factors, /) -> SVDSolveResult:
    problem, policy, plan = prepared.problem, prepared.plan.policy, prepared.plan
    raw_left, raw_right = _stop_arrays(
        _restore(
            prepared, factors.left[:, factors.indices], factors.right[:, factors.indices]
        )
    )
    values = factors.values[factors.indices]

    def unavailable(_: Array) -> tuple[Array, Array, Array, Array, Array, Array]:
        vector = jnp.full_like(values, jnp.nan)
        scalar = jnp.asarray(jnp.nan, values.dtype)
        return vector, vector, vector, scalar, scalar, vector

    evidence = jax.lax.cond(
        (prepared.state.preparation_status == 0) & jnp.all(jnp.isfinite(factors.values)),
        lambda _: triplet_evidence(
            problem, raw_left, values, raw_right, factors.values[0] + factors.eta
        ),
        unavailable,
        jnp.asarray(0),
    )
    (
        left_residual,
        right_residual,
        relative,
        left_orthogonality,
        right_orthogonality,
        energies,
    ) = _stop_arrays(evidence)
    admission = _admission(
        prepared, factors, relative, jnp.maximum(left_orthogonality, right_orthogonality)
    )
    left, values, right, left_response, right_response = _responses(
        prepared, factors, admission.valid
    )
    status = admission.aggregate_status
    if policy.failure.mode == "error":
        message = "SVD solve failed; inspect status-mode diagnostics."
        status = eqx.error_if(status, status != 0, message)
        values = eqx.error_if(values, status != 0, message)
        left, right = (
            eqx.error_if(left, status != 0, message),
            eqx.error_if(right, status != 0, message),
        )
    available = (prepared.state.preparation_status == 0) & jnp.all(
        jnp.isfinite(factors.values)
    )
    rank = rank_evidence(
        factors.values, factors.eta, available, problem, plan, factors.probability
    )
    tail = discarded_singular_maximum(factors.values, policy.count, policy.which)
    spectrum_allowance = compressed_spectrum_allowance(factors.values, problem, plan)
    lower = jnp.maximum(factors.values - spectrum_allowance, 0)
    upper = factors.values + factors.eta + spectrum_allowance
    range_data = SVDRangeEvidence(
        factors.eta,
        factors.allowance,
        spectrum_allowance,
        factors.probability,
        factors.audit_maximum,
        available,
        plan.certificate_kind,
        plan.certificate_kind == "independent-gaussian",
        isinstance(problem.operator, DenseLinearOperator),
        factors.total_energy,
        lower,
        upper,
        factors.eta,
        jnp.hypot(factors.eta, tail + spectrum_allowance),
    )
    converged = (relative <= policy.tolerance.residual) & (
        jnp.maximum(left_orthogonality, right_orthogonality)
        <= policy.tolerance.orthogonality
    )
    diagnostics = SVDSolveDiagnostics(
        left_residual,
        right_residual,
        relative,
        left_orthogonality,
        right_orthogonality,
        admission.isolation,
        admission.boundary,
        admission.pivot_magnitudes,
        admission.pivot_gaps,
        factors.qr_margin,
        energies,
        converged,
        rank,
        jnp.asarray(plan.cost.operator_matvec_count + policy.count, jnp.int32),
        jnp.asarray(plan.cost.adjoint_matvec_count + policy.count, jnp.int32),
    )
    state = prepared.state
    randomized = isinstance(state, RandomizedSVDState)
    route = (
        DerivativeRoute.STOPPED
        if policy.differentiation == "none"
        else DerivativeRoute.UNROLLED
        if randomized
        else DerivativeRoute.SPECTRAL
    )
    provenance = SVDSolveProvenance(
        policy.method.name,
        problem.problem_id,
        plan.plan_id,
        policy.which,
        policy.differentiation,
        route,
        prepared.numeric_version,
        state.root_key if randomized else None,
        state.sketch_version if randomized else jnp.asarray(0, jnp.int32),
        state.audit_version if randomized else jnp.asarray(0, jnp.int32),
    )
    result = SVDSolveResult(
        values,
        _unflatten_columns(problem.operator.target, left),
        _unflatten_columns(problem.operator.source, right),
        left,
        right,
        left_response,
        right_response,
        status,
        admission.primal_status,
        admission.derivative_status,
        admission.valid,
        converged,
        rank,
        range_data,
        admission.leading,
        _stop_arrays(diagnostics),
        provenance,
    )
    return _stop_arrays(result) if policy.differentiation == "none" else result


def _unflatten_columns(space: AbstractVectorSpace, columns: Array, /) -> PyTree[Array]:
    return jax.vmap(space.unflatten, in_axes=1, out_axes=-1)(columns)


def svd(
    problem_or_prepared: SVDProblem | PreparedSVDSolve,
    /,
    *,
    policy: SVDSolvePolicy | SVDSolvePlan | None = None,
    key: PRNGKey | None = None,
) -> SVDSolveResult:
    if isinstance(problem_or_prepared, PreparedSVDSolve):
        if policy is not None or key is not None:
            raise ValueError(
                "policy and replacement key must be omitted for a prepared SVD solve."
            )
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, SVDProblem):
        prepared = prepare_svd(problem_or_prepared, policy, key=key)
    else:
        raise TypeError("Expected an SVDProblem or PreparedSVDSolve.")
    return _assemble(prepared, _factors(prepared))


__all__ = ["plan_svd", "prepare_svd", "refresh_svd", "svd"]
