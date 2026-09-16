#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import (
    AbstractSparseLinearOperator,
    HermitianSpectrum,
    MaterializationPolicy,
    materialize,
)
from .._programming._quadratic import ConvexProgramResult
from ._basis import DenseMomentBasis
from ._problem import _equation_terms, PolynomialOptimizationProblem
from ._relaxation import PreparedPolynomialRelaxation


def _max_abs(value: Array, /) -> Array:
    if value.shape[-1:] == (0,):
        return jnp.asarray(0.0, dtype=value.dtype)
    return jnp.max(jnp.abs(value), axis=-1)


class PolynomialAuditTolerance(StrictModule):
    """Numerical replay tolerances; none of these imply exact arithmetic evidence."""

    feasibility: float = eqx.field(static=True)
    psd: float = eqx.field(static=True)
    rank: float = eqx.field(static=True)
    bound_gap: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        feasibility: float = 1e-7,
        psd: float = 1e-7,
        rank: float = 1e-7,
        bound_gap: float = 1e-6,
    ):
        values = tuple(float(value) for value in (feasibility, psd, rank, bound_gap))
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Polynomial audit tolerances must be finite and positive.")
        self.feasibility, self.psd, self.rank, self.bound_gap = values


class PolynomialCandidateAudit(StrictModule):
    """Independent replay of one point against the original polynomial systems."""

    point: Array
    objective: Array
    equality_values: Array
    inequality_values: Array
    equality_residual: Array
    inequality_violation: Array
    finite: Array
    feasible: Array
    tolerance: float = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)


class PolynomialDualBoundEvidence(StrictModule):
    """Independently replayed conic dual feasibility and its numerical bound."""

    dual: Array
    lower_bound: Array
    stationarity_residual: Array
    cone_residual: Array
    finite: Array
    accepted: Array
    tolerance: float = eqx.field(static=True)


class PolynomialPSDEvidence(StrictModule):
    """Moment/localizing matrices and their independently recomputed PSD residuals."""

    moment_matrix: Array
    localizing_matrices: tuple[Array, ...]
    moment_minimum_eigenvalue: Array
    localizing_minimum_eigenvalues: Array
    moment_psd_residual: Array
    localizing_psd_residuals: Array
    zero_constraint_residual: Array
    finite: Array
    accepted: Array
    tolerance: float = eqx.field(static=True)


class PolynomialFlatnessEvidence(StrictModule):
    """Numerical rank comparison for ``M_r(y)`` and ``M_(r-1)(y)``."""

    moment_rank: Array
    truncated_moment_rank: Array
    moment_eigenvalues: Array
    truncated_moment_eigenvalues: Array
    available: Array
    flat: Array
    tolerance: float = eqx.field(static=True)


class AtomExtractionStatus(IntEnum):
    """Guarded atom-extraction outcome; only rank-one flat extraction is implemented."""

    NOT_ATTEMPTED = 0
    INSUFFICIENT_ORDER = 1
    MOMENT_NOT_ACCEPTED = 2
    NOT_FLAT = 3
    UNSUPPORTED_RANK = 4
    REPLAY_REJECTED = 5
    EXTRACTED = 6


class PolynomialAtomExtraction(StrictModule):
    atoms: Array
    weights: Array
    replay: PolynomialCandidateAudit | None
    status: AtomExtractionStatus = eqx.field(static=True)
    message: str = eqx.field(static=True)


class PolynomialResultStatus(IntEnum):
    """Audit summary without promoting numerical evidence to a proof claim."""

    RELAXATION_REJECTED = 0
    LOWER_BOUND_AVAILABLE = 1
    FEASIBLE_UPPER_BOUND_AVAILABLE = 2
    FLAT_BOUND_MATCH = 3


class PolynomialOptimizationResult(StrictModule):
    """Audited relaxation evidence and replayed original-domain acceptance."""

    moments: Array
    relaxation_value: Array
    lower_bound: Array
    upper_bound: Array
    bound_gap: Array
    provider_path_accepted: Array
    relaxation_constraints_accepted: Array
    lower_bound_available: Array
    upper_bound_available: Array
    original_candidate_accepted: Array
    bound_gap_closed: Array
    exactness_evidenced: Array
    provider_status: Array
    candidate: PolynomialCandidateAudit | None
    dual: PolynomialDualBoundEvidence
    psd: PolynomialPSDEvidence
    flatness: PolynomialFlatnessEvidence
    atom_extraction: PolynomialAtomExtraction
    status: PolynomialResultStatus = eqx.field(static=True)
    prepared_binding_id: str = eqx.field(static=True)


def audit_polynomial_candidate(
    problem: PolynomialOptimizationProblem,
    point: ArrayLike,
    /,
    *,
    tolerance: float = 1e-7,
) -> PolynomialCandidateAudit:
    """Replay a candidate solely against the original polynomial objective/constraints."""

    if not isinstance(problem, PolynomialOptimizationProblem):
        raise TypeError("problem must be a PolynomialOptimizationProblem.")
    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    value = jnp.asarray(point)
    if value.shape != (problem.variable_count,):
        raise ValueError(f"point must have shape ({problem.variable_count},).")
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise TypeError("Polynomial optimization candidates must be real-valued.")
    objective = problem.objective_value(value)
    equalities = problem.equality_values(value)
    inequalities = problem.inequality_values(value)
    equality_residual = _max_abs(equalities)
    inequality_violation = _max_abs(jnp.minimum(inequalities, 0.0))
    equality_accepted = jnp.all(
        jnp.abs(equalities) <= threshold * jnp.maximum(1.0, jnp.abs(equalities))
    )
    inequality_accepted = jnp.all(
        inequalities >= -threshold * jnp.maximum(1.0, jnp.abs(inequalities))
    )
    finite = (
        jnp.all(jnp.isfinite(value))
        & jnp.isfinite(objective)
        & jnp.all(jnp.isfinite(equalities))
        & jnp.all(jnp.isfinite(inequalities))
    )
    feasible = finite & equality_accepted & inequality_accepted
    return PolynomialCandidateAudit(
        value,
        objective,
        equalities,
        inequalities,
        equality_residual,
        inequality_violation,
        finite,
        feasible,
        threshold,
        canonical_fingerprint(
            {
                "kind": "polynomial-candidate",
                "problem": problem.problem_id,
                "point": value,
            }
        ),
        problem.problem_id,
    )


def _dual_bound_evidence(
    prepared: PreparedPolynomialRelaxation,
    result: ConvexProgramResult,
    tolerance: PolynomialAuditTolerance,
    /,
) -> PolynomialDualBoundEvidence:
    program = prepared.program
    dual = jnp.asarray(result.cone_dual, dtype=program.linear.dtype)
    expected = (program.num_constraints,)
    if dual.shape != expected:
        raise ValueError(f"Convex cone dual must have shape {expected}.")
    transpose_action = ein.contract("ji,j->i", program.constraint_matrix, dual)
    stationarity = program.linear + transpose_action
    stationarity_scale = jnp.maximum(
        1.0,
        jnp.maximum(_max_abs(program.linear), _max_abs(transpose_action)),
    )
    stationarity_residual = _max_abs(stationarity) / stationarity_scale
    dual_scale = jnp.maximum(1.0, _max_abs(dual))
    cone_residual = program.cone.dual_residual(dual) / dual_scale
    lower_bound = -ein.contract("i,i->", program.constraint_rhs, dual)
    finite = (
        jnp.all(jnp.isfinite(dual))
        & jnp.isfinite(lower_bound)
        & jnp.isfinite(stationarity_residual)
        & jnp.isfinite(cone_residual)
    )
    accepted = (
        finite
        & (stationarity_residual <= tolerance.feasibility)
        & (cone_residual <= tolerance.psd)
    )
    return PolynomialDualBoundEvidence(
        dual,
        lower_bound,
        stationarity_residual,
        cone_residual,
        finite,
        accepted,
        tolerance.feasibility,
    )


def _psd_evidence(
    prepared: PreparedPolynomialRelaxation,
    moments: Array,
    tolerance: PolynomialAuditTolerance,
    /,
) -> PolynomialPSDEvidence:
    template = prepared.template
    moment_matrix = template.moment_basis.matrix(moments)
    moment_spectrum = HermitianSpectrum(moment_matrix, tolerance=tolerance.rank)
    localizing_matrices: list[Array] = []
    localizing_minima: list[Array] = []
    localizing_residuals: list[Array] = []
    if prepared.problem.inequalities is not None:
        for equation, basis in enumerate(template.localizing_bases):
            _, coefficients = _equation_terms(prepared.problem.inequalities, equation)
            matrix = basis.matrix(moments, coefficients)
            spectrum = HermitianSpectrum(matrix, tolerance=tolerance.rank)
            scale = jnp.maximum(1.0, jnp.max(jnp.abs(spectrum.eigenvalues)))
            localizing_matrices.append(matrix)
            localizing_minima.append(spectrum.minimum_eigenvalue)
            localizing_residuals.append(
                jnp.maximum(-spectrum.minimum_eigenvalue, 0.0) / scale
            )
    dtype = moments.dtype
    minima = (
        jnp.stack(localizing_minima)
        if localizing_minima
        else jnp.empty((0,), dtype=dtype)
    )
    residuals = (
        jnp.stack(localizing_residuals)
        if localizing_residuals
        else jnp.empty((0,), dtype=dtype)
    )
    moment_scale = jnp.maximum(1.0, jnp.max(jnp.abs(moment_spectrum.eigenvalues)))
    moment_residual = jnp.maximum(-moment_spectrum.minimum_eigenvalue, 0.0) / moment_scale
    constraint_matrix = prepared.program.constraint_matrix
    if isinstance(constraint_matrix, AbstractSparseLinearOperator):
        entry_count = prepared.program.num_constraints * prepared.program.num_variables
        dense_constraint_matrix = materialize(
            constraint_matrix,
            MaterializationPolicy(
                max_entries=max(1, entry_count),
                max_bytes=max(1, entry_count * prepared.program.linear.dtype.itemsize),
            ),
        )
    else:
        dense_constraint_matrix = jnp.asarray(constraint_matrix)
    zero_matrix = dense_constraint_matrix[template.zero_slice]
    zero_rhs = prepared.program.constraint_rhs[template.zero_slice]
    zero_values = ein.contract("ij,j->i", zero_matrix, moments) - zero_rhs
    zero_scale = jnp.maximum(1.0, _max_abs(zero_rhs))
    zero_residual = _max_abs(zero_values) / zero_scale
    finite = (
        jnp.all(jnp.isfinite(moment_matrix))
        & jnp.all(jnp.isfinite(minima))
        & jnp.isfinite(zero_residual)
        & moment_spectrum.valid
    )
    localizing_accepted = (
        jnp.all(residuals <= tolerance.psd) if residuals.shape[0] else jnp.asarray(True)
    )
    accepted = (
        finite
        & (moment_residual <= tolerance.psd)
        & localizing_accepted
        & (zero_residual <= tolerance.feasibility)
    )
    return PolynomialPSDEvidence(
        moment_matrix,
        tuple(localizing_matrices),
        moment_spectrum.minimum_eigenvalue,
        minima,
        moment_residual,
        residuals,
        zero_residual,
        finite,
        accepted,
        tolerance.psd,
    )


def _flatness_evidence(
    prepared: PreparedPolynomialRelaxation,
    moments: Array,
    tolerance: PolynomialAuditTolerance,
    psd: PolynomialPSDEvidence,
    /,
) -> PolynomialFlatnessEvidence:
    order = prepared.plan.order
    full = HermitianSpectrum(psd.moment_matrix, tolerance=tolerance.rank)
    if order == 0:
        empty = jnp.empty((0,), dtype=moments.dtype)
        return PolynomialFlatnessEvidence(
            full.numerical_rank,
            jnp.asarray(-1, dtype=jnp.int32),
            full.eigenvalues,
            empty,
            jnp.asarray(False),
            jnp.asarray(False),
            tolerance.rank,
        )
    truncated_basis = DenseMomentBasis(prepared.problem.variable_count, order - 1)
    truncated_moments = moments[: truncated_basis.moment_count]
    truncated_matrix = truncated_basis.matrix(truncated_moments)
    truncated = HermitianSpectrum(truncated_matrix, tolerance=tolerance.rank)
    available = full.valid & truncated.valid & psd.accepted
    flat = available & (full.numerical_rank == truncated.numerical_rank)
    return PolynomialFlatnessEvidence(
        full.numerical_rank,
        truncated.numerical_rank,
        full.eigenvalues,
        truncated.eigenvalues,
        available,
        flat,
        tolerance.rank,
    )


def _empty_extraction(
    variable_count: int,
    dtype: Any,
    status: AtomExtractionStatus,
    message: str,
    /,
    replay: PolynomialCandidateAudit | None = None,
) -> PolynomialAtomExtraction:
    return PolynomialAtomExtraction(
        jnp.empty((0, variable_count), dtype=dtype),
        jnp.empty((0,), dtype=dtype),
        replay,
        status,
        message,
    )


def _extract_rank_one_atom(
    prepared: PreparedPolynomialRelaxation,
    moments: Array,
    relaxation_accepted: Array,
    flatness: PolynomialFlatnessEvidence,
    tolerance: PolynomialAuditTolerance,
    /,
) -> PolynomialAtomExtraction:
    variables = prepared.problem.variable_count
    if not bool(np.asarray(relaxation_accepted)):
        return _empty_extraction(
            variables,
            moments.dtype,
            AtomExtractionStatus.MOMENT_NOT_ACCEPTED,
            "independent moment/localizing audit did not accept the relaxation point",
        )
    if prepared.plan.order == 0:
        return _empty_extraction(
            variables,
            moments.dtype,
            AtomExtractionStatus.INSUFFICIENT_ORDER,
            "order zero has no first moments from which to recover a point",
        )
    if not bool(np.asarray(flatness.flat)):
        return _empty_extraction(
            variables,
            moments.dtype,
            AtomExtractionStatus.NOT_FLAT,
            "moment rank did not satisfy the checked flat-extension condition",
        )
    if int(np.asarray(flatness.moment_rank)) != 1:
        return _empty_extraction(
            variables,
            moments.dtype,
            AtomExtractionStatus.UNSUPPORTED_RANK,
            "only rank-one flat moment extraction is implemented",
        )
    basis = prepared.template.moment_basis.moments
    normalization = moments[basis.index((0,) * variables)]
    coordinates = jnp.stack(
        tuple(
            moments[
                basis.index(
                    tuple(1 if axis == current else 0 for axis in range(variables))
                )
            ]
            / normalization
            for current in range(variables)
        )
    )
    replay = audit_polynomial_candidate(
        prepared.problem,
        coordinates,
        tolerance=tolerance.feasibility,
    )
    if not bool(np.asarray(replay.feasible)):
        return _empty_extraction(
            variables,
            moments.dtype,
            AtomExtractionStatus.REPLAY_REJECTED,
            "rank-one recovered point failed original polynomial replay",
            replay,
        )
    return PolynomialAtomExtraction(
        coordinates[None, :],
        jnp.asarray([normalization], dtype=moments.dtype),
        replay,
        AtomExtractionStatus.EXTRACTED,
        "one rank-one atom passed original polynomial replay",
    )


def audit_polynomial_relaxation(
    prepared: PreparedPolynomialRelaxation,
    result: ConvexProgramResult,
    /,
    *,
    candidate: ArrayLike | None = None,
    tolerance: PolynomialAuditTolerance | None = None,
) -> PolynomialOptimizationResult:
    """Audit provider output, conic moments, and original-domain candidates separately."""

    if not isinstance(prepared, PreparedPolynomialRelaxation):
        raise TypeError("prepared must be a PreparedPolynomialRelaxation.")
    if not isinstance(result, ConvexProgramResult):
        raise TypeError("result must be a ConvexProgramResult.")
    if result.provenance.structure_id != prepared.program.structure_id:
        raise ValueError(
            "Convex result does not match the prepared relaxation structure."
        )
    moments = jnp.asarray(result.primal)
    expected = (prepared.template.moment_basis.moment_count,)
    if moments.shape != expected:
        raise ValueError(f"Convex primal moments must have shape {expected}.")
    policy = PolynomialAuditTolerance() if tolerance is None else tolerance
    if not isinstance(policy, PolynomialAuditTolerance):
        raise TypeError("tolerance must be PolynomialAuditTolerance or None.")

    provider_accepted = jnp.asarray(result.successful)
    dual = _dual_bound_evidence(prepared, result, policy)
    psd = _psd_evidence(prepared, moments, policy)
    relaxation_accepted = provider_accepted & psd.accepted
    lower_available = provider_accepted & dual.accepted
    flatness = _flatness_evidence(prepared, moments, policy, psd)
    relaxation_value = ein.contract("i,i->", prepared.program.linear, moments)
    nan = jnp.asarray(jnp.nan, dtype=relaxation_value.dtype)
    lower_bound = jnp.where(lower_available, dual.lower_bound, nan)
    extraction = _extract_rank_one_atom(
        prepared,
        moments,
        relaxation_accepted,
        flatness,
        policy,
    )
    submitted = (
        None
        if candidate is None
        else audit_polynomial_candidate(
            prepared.problem,
            candidate,
            tolerance=policy.feasibility,
        )
    )
    submitted_available = jnp.asarray(False) if submitted is None else submitted.feasible
    extracted_available = (
        jnp.asarray(False)
        if extraction.replay is None
        else extraction.replay.feasible
        & (extraction.status == AtomExtractionStatus.EXTRACTED)
    )
    submitted_objective = nan if submitted is None else submitted.objective
    extracted_objective = (
        nan if extraction.replay is None else extraction.replay.objective
    )
    upper_bound = jnp.where(
        submitted_available & extracted_available,
        jnp.minimum(submitted_objective, extracted_objective),
        jnp.where(
            submitted_available,
            submitted_objective,
            jnp.where(extracted_available, extracted_objective, nan),
        ),
    )
    upper_available = submitted_available | extracted_available
    gap = jnp.where(
        lower_available & upper_available,
        upper_bound - lower_bound,
        nan,
    )
    gap_scale = jnp.maximum(
        1.0,
        jnp.maximum(jnp.abs(lower_bound), jnp.abs(upper_bound)),
    )
    gap_closed = (
        lower_available
        & upper_available
        & jnp.isfinite(gap)
        & (jnp.abs(gap) <= policy.bound_gap * gap_scale)
    )
    exactness_evidenced = gap_closed & flatness.flat

    if bool(np.asarray(exactness_evidenced)):
        status = PolynomialResultStatus.FLAT_BOUND_MATCH
    elif bool(np.asarray(upper_available)):
        status = PolynomialResultStatus.FEASIBLE_UPPER_BOUND_AVAILABLE
    elif bool(np.asarray(lower_available)):
        status = PolynomialResultStatus.LOWER_BOUND_AVAILABLE
    else:
        status = PolynomialResultStatus.RELAXATION_REJECTED
    return PolynomialOptimizationResult(
        moments,
        relaxation_value,
        lower_bound,
        upper_bound,
        gap,
        provider_accepted,
        psd.accepted,
        lower_available,
        upper_available,
        submitted_available,
        gap_closed,
        exactness_evidenced,
        result.status,
        submitted,
        dual,
        psd,
        flatness,
        extraction,
        status,
        prepared.binding_id,
    )


__all__ = [
    "AtomExtractionStatus",
    "PolynomialAtomExtraction",
    "PolynomialAuditTolerance",
    "PolynomialCandidateAudit",
    "PolynomialDualBoundEvidence",
    "PolynomialFlatnessEvidence",
    "PolynomialOptimizationResult",
    "PolynomialPSDEvidence",
    "PolynomialResultStatus",
    "audit_polynomial_candidate",
    "audit_polynomial_relaxation",
]
