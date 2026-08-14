#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._costs import LinearCostEstimate
from ._operators import (
    AbstractLinearOperator,
    AdjointLinearOperator,
    BlockLinearOperator,
    ComposedLinearOperator,
    DenseLinearOperator,
    DiagonalLinearOperator,
    IdentityLinearOperator,
    ScaledLinearOperator,
    SumLinearOperator,
    TransposeLinearOperator,
)
from ._policies import (
    AbstractLinearMethod,
    AutoLinearMethod,
    BiCGStab,
    ConjugateGradient,
    DenseCholesky,
    DenseLU,
    DenseQR,
    DenseSVD,
    FGMRES,
    GeneralizedLSMR,
    GMRES,
    HostSparseLU,
    LinearSolvePolicy,
    LSMR,
    MINRES,
    PCG,
    SparseDirect,
    StructuredDirect,
)
from ._problems import (
    _problem_structure,
    AbstractLinearProblem,
    LeastSquaresProblem,
    LinearSystem,
    MinimumNormProblem,
)
from ._spaces import _coordinate_dtype, _has_diagonal_pairing, _has_euclidean_pairing
from ._sparse_contract import AbstractSparseLinearOperator
from ._structured_operators import (
    _is_structured_exact,
    BandedLinearOperator,
    BlockDiagonalLinearOperator,
    DiagonalPlusLowRankLinearOperator,
    KroneckerLinearOperator,
    KroneckerSumLinearOperator,
    LowRankLinearOperator,
    PermutationLinearOperator,
    SymmetricLowRankLinearOperator,
    TriangularLinearOperator,
    TridiagonalLinearOperator,
)


LinearBackend: TypeAlias = Literal[
    "jax-structured",
    "jax-dense",
    "jax-sparse",
    "host-sparse",
    "native-krylov",
    "matfree",
    "lineax",
]


class LinearSolvePlan(StrictModule):
    """Immutable symbolic selection; all numerical state belongs to preparation."""

    policy: LinearSolvePolicy
    candidates: tuple[LinearCostEstimate, ...]
    problem_id: str = eqx.field(static=True)
    problem_kind: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    problem_signature: str = eqx.field(static=True)
    backend: LinearBackend = eqx.field(static=True)
    method: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    rejected: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: AbstractLinearProblem,
        policy: LinearSolvePolicy,
        backend: LinearBackend,
        method: str,
        reason: str,
        rejected: tuple[str, ...] = (),
        candidates: tuple[LinearCostEstimate, ...] = (),
    ):
        if backend not in (
            "jax-structured",
            "jax-dense",
            "jax-sparse",
            "host-sparse",
            "native-krylov",
            "matfree",
            "lineax",
        ):
            raise ValueError("Unknown linear backend.")
        values = (str(method), str(reason))
        if any(not value for value in values):
            raise ValueError("Plan method and reason must be non-empty.")
        self.policy = policy
        self.candidates = candidates
        self.problem_id = problem.problem_id
        self.problem_kind = problem.kind
        self.operator_id = problem.operator.operator_id
        self.problem_signature = _problem_structure(problem)
        self.backend = backend
        self.method, self.reason = values
        self.rejected = tuple(str(value) for value in rejected)
        self.plan_id = canonical_fingerprint(
            {
                "problem": problem.problem_id,
                "kind": problem.kind,
                "problem_signature": self.problem_signature,
                "backend": backend,
                "method": _method_configuration(method, policy),
                "operator": problem.operator.operator_id,
                "source": problem.operator.source.space_id,
                "target": problem.operator.target.space_id,
                "batch_shape": list(problem.operator.batch_shape),
                "properties": {
                    "diagonal": problem.operator.properties.diagonal,
                    "triangular": problem.operator.properties.triangular,
                    "self_adjoint": problem.operator.properties.self_adjoint,
                    "positive_definite": problem.operator.properties.positive_definite,
                    "positive_semidefinite": (
                        problem.operator.properties.positive_semidefinite
                    ),
                    "block_diagonal": problem.operator.properties.block_diagonal,
                    "rank": problem.operator.properties.rank,
                    "evidence": [
                        list(item) for item in problem.operator.properties.evidence
                    ],
                },
                "capabilities": {
                    "transpose": problem.operator.capabilities.transpose,
                    "adjoint": problem.operator.capabilities.adjoint,
                    "materialize": problem.operator.capabilities.materialize,
                },
                "rank_cutoff": policy.rank.relative_cutoff,
                "require_full_rank": policy.rank.require_full_rank,
                "relative_tolerance": policy.tolerance.relative,
                "absolute_tolerance": policy.tolerance.absolute,
                "max_steps": policy.tolerance.max_steps,
                "preconditioner": (
                    None
                    if policy.preconditioner is None
                    else policy.preconditioner.preconditioner_id
                ),
                "differentiation": policy.differentiation.mode,
                "failure": policy.failure.mode,
                "materialization": {
                    "max_entries": policy.materialization.max_entries,
                    "max_bytes": policy.materialization.max_bytes,
                },
                "resources": {
                    "factorization_bytes": policy.resources.factorization_bytes,
                    "workspace_bytes": policy.resources.workspace_bytes,
                    "krylov_basis_bytes": policy.resources.krylov_basis_bytes,
                },
            }
        )


def plan(
    problem: AbstractLinearProblem,
    policy: LinearSolvePolicy | None = None,
    /,
) -> LinearSolvePlan:
    """Select the first feasible candidate in deterministic capability order."""
    if not isinstance(problem, AbstractLinearProblem):
        raise TypeError("problem must be an AbstractLinearProblem.")
    policy_ = LinearSolvePolicy() if policy is None else policy
    if not isinstance(policy_, LinearSolvePolicy):
        raise TypeError("policy must be a LinearSolvePolicy.")
    requested = policy_.method
    selected, reason, rejected = (
        _auto_method(problem, policy_)
        if isinstance(requested, AutoLinearMethod)
        else (requested, "explicit policy", ())
    )
    backend = _validate_method(problem, selected, policy_)
    selected_estimate = _selected_estimate(
        problem,
        selected,
        backend,
        policy_,
        reason,
    )
    _require_selected_resources(selected_estimate, policy_)
    estimates = tuple(_rejected_estimate(entry) for entry in rejected) + (
        selected_estimate,
    )
    return LinearSolvePlan(
        problem=problem,
        policy=policy_,
        backend=backend,
        method=selected.name,
        reason=reason,
        rejected=rejected,
        candidates=estimates,
    )


def _auto_method(
    problem: AbstractLinearProblem,
    policy: LinearSolvePolicy,
    /,
) -> tuple[AbstractLinearMethod, str, tuple[str, ...]]:
    operator = problem.operator
    rejected: list[str] = []
    if isinstance(problem, LinearSystem) and _is_structured_exact(operator):
        fits, explanation = _structured_candidate_fits(problem, policy)
        if fits:
            return StructuredDirect(), explanation, ()
        rejected.append(f"structured-direct: {explanation}")

    explicit = _is_explicit_operator(operator)
    if isinstance(problem, LinearSystem) and isinstance(
        operator, AbstractSparseLinearOperator
    ):
        if (
            _cuda_sparse_available()
            and policy.preconditioner is None
            and policy.rank.relative_cutoff is None
        ):
            return SparseDirect(), "canonical CSR with native CUDA sparse QR", ()
        rejected.append(
            "sparse-direct: execution requires CUDA, no preconditioner, and no "
            "numerical rank cutoff"
        )
    if isinstance(problem, LinearSystem):
        dense_method: AbstractLinearMethod
        if _certified_positive_definite(operator) and _has_diagonal_pairing(
            operator.source
        ):
            dense_method = DenseCholesky()
        else:
            dense_method = DenseLU()
        if explicit and policy.preconditioner is None:
            fits, explanation = _dense_candidate_fits(problem, dense_method, policy)
            if fits:
                return dense_method, explanation, ()
            rejected.append(f"{dense_method.name}: {explanation}")
        elif explicit:
            rejected.append(
                f"{dense_method.name}: dense direct execution does not accept "
                "preconditioners"
            )
        preconditioner_is_positive = (
            policy.preconditioner is None or policy.preconditioner.positive_definite
        )
        if _certified_positive_definite(operator) and preconditioner_is_positive:
            return PCG(), "positive-definite Krylov fallback", tuple(rejected)
        if _certified_self_adjoint(operator) and preconditioner_is_positive:
            return MINRES(), "self-adjoint indefinite Krylov fallback", tuple(rejected)
        if policy.preconditioner is not None:
            return (
                FGMRES(),
                "general variable-preconditioned Krylov fallback",
                tuple(rejected),
            )
        if policy.differentiation.mode == "algorithmic" or not _has_diagonal_pairing(
            operator.source
        ):
            return (
                FGMRES(),
                "pairing-aware native Krylov fallback",
                tuple(rejected),
            )
        return GMRES(), "general square Krylov fallback", tuple(rejected)

    if isinstance(problem, (LeastSquaresProblem, MinimumNormProblem)):
        if explicit and policy.preconditioner is None:
            fits, explanation = _dense_candidate_fits(problem, DenseSVD(), policy)
            if fits:
                return DenseSVD(), explanation, ()
            rejected.append(f"dense-svd: {explanation}")
        elif explicit:
            rejected.append(
                "dense-svd: dense direct execution does not accept preconditioners"
            )
        if _matfree_lsmr_eligible(problem, policy):
            return LSMR(), "real-Euclidean Matfree LSMR envelope", tuple(rejected)
        return (
            GeneralizedLSMR(),
            "pairing-aware generalized least-squares Krylov fallback",
            tuple(rejected),
        )
    raise TypeError(f"Unsupported problem type {type(problem).__name__}.")


def _validate_method(
    problem: AbstractLinearProblem,
    method: AbstractLinearMethod,
    policy: LinearSolvePolicy,
    /,
) -> LinearBackend:
    operator = problem.operator
    preconditioner = policy.preconditioner
    if preconditioner is not None and not preconditioner.space.compatible(
        operator.source
    ):
        raise ValueError("Preconditioner space must match the operator source.")
    if isinstance(method, StructuredDirect):
        if not isinstance(problem, LinearSystem) or not _is_structured_exact(operator):
            raise ValueError(
                "StructuredDirect requires a LinearSystem with recognized exact structure."
            )
        if preconditioner is not None:
            raise ValueError("StructuredDirect does not accept preconditioners.")
        if policy.rank.relative_cutoff is not None:
            raise ValueError("StructuredDirect cannot enforce a numerical rank cutoff.")
        if policy.rank.require_full_rank and not _certifies_full_rank(operator):
            raise ValueError(
                "StructuredDirect full-rank requirements need a full-rank certificate."
            )
        fits, explanation = _structured_candidate_fits(problem, policy)
        if not fits:
            raise ValueError(f"Selected structured method is infeasible: {explanation}.")
        return "jax-structured"
    if isinstance(method, (DenseLU, DenseCholesky)):
        if preconditioner is not None:
            raise ValueError("Dense direct methods do not accept preconditioners.")
        if policy.rank.relative_cutoff is not None:
            raise ValueError(
                "Dense square direct methods cannot enforce a numerical rank cutoff."
            )
        if not isinstance(problem, LinearSystem):
            raise ValueError(f"{method.name} requires a LinearSystem.")
        if isinstance(method, DenseCholesky) and not _certified_positive_definite(
            operator
        ):
            raise ValueError("Dense Cholesky requires certified positive definiteness.")
        if isinstance(method, DenseCholesky) and not _has_diagonal_pairing(
            operator.source
        ):
            raise ValueError(
                "Dense Cholesky requires a Euclidean or diagonal source pairing."
            )
        _require_dense_candidate(problem, method, policy)
        return "jax-dense"
    if isinstance(method, (DenseQR, DenseSVD)):
        if preconditioner is not None:
            raise ValueError("Dense rectangular methods do not accept preconditioners.")
        if not isinstance(problem, (LeastSquaresProblem, MinimumNormProblem)):
            raise ValueError(
                f"{method.name} requires least-squares or minimum-norm semantics."
            )
        if isinstance(method, DenseQR) and isinstance(problem, MinimumNormProblem):
            raise ValueError("Dense QR does not implement minimum-norm semantics.")
        if not _dense_metric_pairings_supported(problem):
            raise ValueError(
                "Dense rectangular methods require Euclidean or diagonal metric pairings."
            )
        _require_dense_candidate(problem, method, policy)
        if isinstance(method, DenseQR):
            assert isinstance(problem, LeastSquaresProblem)
            rows = operator.target.size + (
                0 if problem.regularizer is None else problem.regularizer.target.size
            )
            if rows < operator.source.size:
                raise ValueError("Dense QR requires at least as many rows as columns.")
        return "jax-dense"
    if isinstance(method, (SparseDirect, HostSparseLU)):
        if not isinstance(problem, LinearSystem):
            raise ValueError(f"{method.name} requires a LinearSystem.")
        if not isinstance(operator, AbstractSparseLinearOperator):
            raise ValueError(f"{method.name} requires canonical sparse storage.")
        if policy.rank.relative_cutoff is not None:
            raise ValueError(
                "Sparse direct methods cannot enforce a numerical rank cutoff."
            )
        if operator.source.size != operator.target.size:
            raise ValueError(f"{method.name} requires a square operator.")
        if preconditioner is not None:
            raise ValueError("Sparse direct methods do not accept preconditioners.")
        if isinstance(method, SparseDirect):
            if not _cuda_sparse_available():
                raise ValueError(
                    "SparseDirect requires CUDA; use HostSparseLU explicitly for "
                    "the non-JIT CPU fallback."
                )
            if policy.differentiation.mode == "algorithmic":
                raise ValueError(
                    "SparseDirect exposes mathematical differentiation, not an "
                    "algorithmic QR derivative."
                )
            return "jax-sparse"
        if policy.differentiation.mode != "none":
            raise ValueError(
                "HostSparseLU is non-JIT and requires DifferentiationPolicy('none')."
            )
        return "host-sparse"
    if operator.batch_shape:
        raise ValueError("Iterative providers require explicit batched execution policy.")
    square_iterative = isinstance(
        method,
        (PCG, MINRES, FGMRES, ConjugateGradient, GMRES, BiCGStab),
    )
    if (
        square_iterative
        and isinstance(problem, LinearSystem)
        and not operator.source.compatible(operator.target)
    ):
        raise ValueError(
            "Iterative LinearSystem methods require compatible source and target "
            "spaces; use a direct method or represent the map as an endomorphism."
        )
    if policy.rank.relative_cutoff is not None:
        raise ValueError("Iterative providers cannot enforce a numerical rank cutoff.")
    if policy.rank.require_full_rank and not _certifies_full_rank(operator):
        raise ValueError(
            "Iterative full-rank requirements need a full-rank operator certificate."
        )
    if isinstance(method, PCG):
        if not isinstance(problem, LinearSystem):
            raise ValueError("PCG requires a LinearSystem.")
        if not _certified_positive_definite(operator):
            raise ValueError("PCG requires certified positive definiteness.")
        if preconditioner is not None and not preconditioner.positive_definite:
            raise ValueError("PCG requires a certified positive-definite preconditioner.")
        return "native-krylov"
    if isinstance(method, MINRES):
        if not isinstance(problem, LinearSystem):
            raise ValueError("MINRES requires a LinearSystem.")
        if not _certified_self_adjoint(operator):
            raise ValueError("MINRES requires certified self-adjoint structure.")
        if preconditioner is not None and not preconditioner.positive_definite:
            raise ValueError("MINRES requires a positive-definite preconditioner.")
        return "native-krylov"
    if isinstance(method, FGMRES):
        if not isinstance(problem, LinearSystem):
            raise ValueError("FGMRES requires a LinearSystem.")
        return "native-krylov"
    if isinstance(method, GeneralizedLSMR):
        if not isinstance(problem, (LeastSquaresProblem, MinimumNormProblem)):
            raise ValueError("GeneralizedLSMR requires least-squares semantics.")
        if preconditioner is not None:
            raise ValueError(
                "GeneralizedLSMR uses problem transforms, not solve preconditioners."
            )
        if not operator.capabilities.adjoint:
            raise ValueError("GeneralizedLSMR requires an explicit adjoint capability.")
        return "native-krylov"
    if isinstance(method, LSMR):
        if not _matfree_lsmr_eligible(problem, policy):
            raise ValueError(
                "Matfree LSMR requires real Euclidean spaces, no preconditioner, "
                "and no weighted or explicit regularized residual."
            )
        return "matfree"
    if isinstance(method, ConjugateGradient):
        if not isinstance(problem, LinearSystem):
            raise ValueError("CG requires a LinearSystem.")
        if not _has_diagonal_pairing(operator.source):
            raise ValueError(
                "Lineax methods require a Euclidean or diagonal source pairing; "
                "use FGMRES for a general Hilbert pairing."
            )
        if not _certified_positive_definite(operator) or not _is_real(operator):
            raise ValueError(
                "Lineax CG requires a real certified positive-definite operator."
            )
        if preconditioner is not None and not preconditioner.positive_definite:
            raise ValueError("CG requires a certified positive-definite preconditioner.")
        _reject_algorithmic_lineax(policy)
        return "lineax"
    if isinstance(method, GMRES):
        if not isinstance(problem, LinearSystem):
            raise ValueError("GMRES requires a LinearSystem.")
        if not _has_diagonal_pairing(operator.source):
            raise ValueError(
                "GMRES requires a Euclidean or diagonal source pairing; "
                "use FGMRES for a general Hilbert pairing."
            )
        return "native-krylov"
    if isinstance(method, BiCGStab):
        if not isinstance(problem, LinearSystem):
            raise ValueError("bicgstab requires a LinearSystem.")
        if not _has_diagonal_pairing(operator.source):
            raise ValueError(
                "Lineax methods require a Euclidean or diagonal source pairing."
            )
        _reject_algorithmic_lineax(policy)
        return "lineax"
    raise TypeError(f"Unsupported linear method {type(method).__name__}.")


def _reject_algorithmic_lineax(policy: LinearSolvePolicy, /) -> None:
    if policy.differentiation.mode == "algorithmic":
        raise ValueError(
            "Lineax does not expose the executed finite iteration derivative."
        )


def _matfree_lsmr_eligible(
    problem: AbstractLinearProblem,
    policy: LinearSolvePolicy,
    /,
) -> bool:
    if not isinstance(problem, (LeastSquaresProblem, MinimumNormProblem)):
        return False
    if policy.preconditioner is not None or policy.differentiation.mode == "algorithmic":
        return False
    if not _is_real(problem.operator):
        return False
    if not _has_euclidean_pairing(problem.operator.source) or not _has_euclidean_pairing(
        problem.operator.target
    ):
        return False
    return not (
        isinstance(problem, LeastSquaresProblem)
        and (problem.weights is not None or problem.regularizer is not None)
    )


def _dense_metric_pairings_supported(problem: AbstractLinearProblem, /) -> bool:
    if isinstance(problem, (LinearSystem, MinimumNormProblem)):
        return _has_diagonal_pairing(problem.operator.source)
    if isinstance(problem, LeastSquaresProblem):
        return _has_diagonal_pairing(problem.operator.target) and (
            problem.regularizer is None
            or _has_diagonal_pairing(problem.regularizer.target)
        )
    return False


def _structured_dense_entries(operator: AbstractLinearOperator, /) -> int:
    if isinstance(operator, BandedLinearOperator):
        return operator.source.size * operator.target.size
    if isinstance(operator, DiagonalPlusLowRankLinearOperator):
        return (
            0
            if operator.nonsingular_diagonal
            else operator.source.size * operator.target.size
        )
    if isinstance(operator, BlockDiagonalLinearOperator):
        return max(
            (_structured_dense_entries(block) for block in operator.blocks),
            default=0,
        )
    if isinstance(operator, KroneckerLinearOperator):
        return max(
            (_structured_dense_entries(factor) for factor in operator.factors),
            default=0,
        )
    return 0


def _structured_factorization_entries(
    operator: AbstractLinearOperator,
    /,
) -> int:
    if isinstance(operator, (DenseLinearOperator, BandedLinearOperator)):
        dimension = operator.source.size
        return dimension * dimension + dimension
    if isinstance(operator, TridiagonalLinearOperator):
        return 5 * operator.source.size
    if isinstance(operator, DiagonalPlusLowRankLinearOperator):
        dimension = operator.source.size
        rank = operator.left_factor.shape[1]
        core_lu = rank * rank + rank
        woodbury = core_lu + dimension * rank + dimension
        if operator.nonsingular_diagonal:
            return woodbury
        dense_lu = dimension * dimension + dimension
        return dense_lu + woodbury
    if isinstance(operator, BlockDiagonalLinearOperator):
        return sum(_structured_factorization_entries(block) for block in operator.blocks)
    if isinstance(operator, KroneckerLinearOperator):
        return sum(
            _structured_factorization_entries(factor) for factor in operator.factors
        )
    return 0


def _structured_candidate_fits(
    problem: AbstractLinearProblem,
    policy: LinearSolvePolicy,
    /,
) -> tuple[bool, str]:
    operator = problem.operator
    if policy.preconditioner is not None:
        return False, "structured direct execution does not accept preconditioners"
    if policy.rank.relative_cutoff is not None:
        return False, "structured direct execution cannot enforce a rank cutoff"
    if policy.rank.require_full_rank and not _certifies_full_rank(operator):
        return False, "full-rank execution lacks a full-rank certificate"
    materialized_entries = _structured_dense_entries(operator)
    factor_entries = _structured_factorization_entries(operator)
    itemsize = _coordinate_dtype(operator.source).itemsize
    materialized_bytes = materialized_entries * itemsize
    factor_bytes = factor_entries * itemsize
    if materialized_entries > policy.materialization.max_entries:
        return False, (
            f"fallback materialization requires {materialized_entries} entries"
        )
    if materialized_bytes > policy.materialization.max_bytes:
        return False, (f"fallback materialization requires {materialized_bytes} bytes")
    if factor_bytes > policy.resources.factorization_bytes:
        return False, f"factorization estimate {factor_bytes} exceeds budget"
    workspace = max(
        (operator.source.size + operator.target.size) * itemsize,
        factor_bytes,
    )
    if workspace > policy.resources.workspace_bytes:
        return False, f"workspace estimate {workspace} exceeds budget"
    return True, "recognized exact structure fits declared budgets"


def _dense_candidate_fits(
    problem: AbstractLinearProblem,
    method: AbstractLinearMethod,
    policy: LinearSolvePolicy,
    /,
) -> tuple[bool, str]:
    operator = problem.operator
    if isinstance(method, DenseCholesky) and not _has_diagonal_pairing(operator.source):
        return False, "Cholesky requires a coordinate-diagonal source pairing"
    if isinstance(method, (DenseQR, DenseSVD)) and not _dense_metric_pairings_supported(
        problem
    ):
        return False, "rectangular dense factors require coordinate-diagonal metrics"
    if not operator.capabilities.materialize:
        return False, "operator does not declare materialization"
    rows, columns = operator.target.size, operator.source.size
    if isinstance(problem, LeastSquaresProblem) and problem.regularizer is not None:
        rows += problem.regularizer.target.size
        if not problem.regularizer.capabilities.materialize:
            return False, "regularizer does not declare materialization"
    itemsize = _coordinate_dtype(operator.source).itemsize
    batch_count = prod(operator.batch_shape or (1,))
    requires_materialization = _requires_dense_materialization(problem)
    additional = (
        batch_count * rows * columns * itemsize if requires_materialization else 0
    )
    entries = batch_count * rows * columns
    if additional and entries > policy.materialization.max_entries:
        return False, f"materialization requires {entries} entries"
    if additional > policy.materialization.max_bytes:
        return False, f"materialization requires {additional} additional bytes"
    factor = batch_count * _factorization_bytes(method, rows, columns, itemsize)
    if factor > policy.resources.factorization_bytes:
        return False, f"factorization estimate {factor} exceeds budget"
    workspace = batch_count * rows * columns * itemsize
    if workspace > policy.resources.workspace_bytes:
        return False, f"workspace estimate {workspace} exceeds budget"
    return True, f"dense factors fit declared budgets ({factor} factor bytes)"


def _require_dense_candidate(
    problem: AbstractLinearProblem,
    method: AbstractLinearMethod,
    policy: LinearSolvePolicy,
    /,
) -> None:
    fits, reason = _dense_candidate_fits(problem, method, policy)
    if not fits:
        raise ValueError(f"Selected dense method is infeasible: {reason}.")


def _factorization_bytes(
    method: AbstractLinearMethod,
    rows: int,
    columns: int,
    itemsize: int,
    /,
) -> int:
    rank = min(rows, columns)
    if isinstance(method, (DenseLU, DenseCholesky)):
        return columns * columns * itemsize
    if isinstance(method, DenseQR):
        return (rows * columns + columns * columns) * itemsize
    if isinstance(method, DenseSVD):
        return (rows * rank + rank + rank * columns) * itemsize
    return 0


def _requires_dense_materialization(problem: AbstractLinearProblem, /) -> bool:
    if not _stores_dense_matrix(problem.operator):
        return True
    return isinstance(problem, LeastSquaresProblem) and (
        problem.regularizer is not None and not _stores_dense_matrix(problem.regularizer)
    )


def _stores_dense_matrix(operator: AbstractLinearOperator, /) -> bool:
    if isinstance(operator, (DenseLinearOperator, TriangularLinearOperator)):
        return True
    if isinstance(operator, (TransposeLinearOperator, AdjointLinearOperator)):
        return _stores_dense_matrix(operator.operator)
    return False


def _array_bytes(*values: jax.Array) -> int:
    return sum(int(value.size * value.dtype.itemsize) for value in values)


def _existing_storage_bytes(operator: AbstractLinearOperator, /) -> int:
    if isinstance(operator, DenseLinearOperator):
        return _array_bytes(operator.matrix)
    if isinstance(operator, DiagonalLinearOperator):
        return _array_bytes(operator.diagonal)
    if isinstance(operator, PermutationLinearOperator):
        return _array_bytes(operator.permutation, operator.inverse_permutation)
    if isinstance(operator, TriangularLinearOperator):
        return _array_bytes(operator.matrix)
    if isinstance(operator, TridiagonalLinearOperator):
        return _array_bytes(operator.lower, operator.diagonal, operator.upper)
    if isinstance(operator, BandedLinearOperator):
        return _array_bytes(operator.bands)
    if isinstance(operator, LowRankLinearOperator):
        return _array_bytes(operator.left_factor, operator.right_factor)
    if isinstance(operator, SymmetricLowRankLinearOperator):
        return _array_bytes(operator.factor, operator.weights)
    if isinstance(operator, DiagonalPlusLowRankLinearOperator):
        return _array_bytes(
            operator.diagonal,
            operator.left_factor,
            operator.right_factor,
        )
    if isinstance(operator, (BlockDiagonalLinearOperator, KroneckerLinearOperator)):
        children = (
            operator.blocks
            if isinstance(operator, BlockDiagonalLinearOperator)
            else operator.factors
        )
        return sum(_existing_storage_bytes(child) for child in children)
    if isinstance(operator, KroneckerSumLinearOperator):
        return sum(_existing_storage_bytes(factor) for factor in operator.factors)
    if isinstance(operator, AbstractSparseLinearOperator):
        storage = operator.sparse_storage()
        return _array_bytes(storage.values, storage.indices, storage.indptr)
    if isinstance(operator, (TransposeLinearOperator, AdjointLinearOperator)):
        return _existing_storage_bytes(operator.operator)
    if isinstance(operator, ScaledLinearOperator):
        return _existing_storage_bytes(operator.operator) + _array_bytes(operator.scalar)
    if isinstance(operator, (SumLinearOperator, ComposedLinearOperator)):
        return _existing_storage_bytes(operator.left) + _existing_storage_bytes(
            operator.right
        )
    if isinstance(operator, BlockLinearOperator):
        return sum(
            _existing_storage_bytes(block)
            for row in operator.blocks
            for block in row
            if block is not None
        )
    return 0


def _selected_estimate(
    problem: AbstractLinearProblem,
    method: AbstractLinearMethod,
    backend: LinearBackend,
    policy: LinearSolvePolicy,
    reason: str,
    /,
) -> LinearCostEstimate:
    rows, columns = problem.operator.target.size, problem.operator.source.size
    if isinstance(problem, LeastSquaresProblem) and problem.regularizer is not None:
        rows += problem.regularizer.target.size
    itemsize = _coordinate_dtype(problem.operator.source).itemsize
    existing = _existing_storage_bytes(problem.operator)
    batch_count = prod(problem.operator.batch_shape or (1,))
    dense_direct = backend == "jax-dense"
    if isinstance(problem, LeastSquaresProblem) and problem.regularizer is not None:
        existing += _existing_storage_bytes(problem.regularizer)
    sparse_direct = backend in ("jax-sparse", "host-sparse")
    structured_direct = backend == "jax-structured"
    dense_requires_materialization = dense_direct and _requires_dense_materialization(
        problem
    )
    structured_dense_bytes = (
        _structured_dense_entries(problem.operator) * itemsize if structured_direct else 0
    )
    structured_factor_bytes = (
        _structured_factorization_entries(problem.operator) * itemsize
        if structured_direct
        else 0
    )
    iterative = backend in ("lineax", "native-krylov", "matfree")
    if dense_direct:
        factorization_bytes = batch_count * _factorization_bytes(
            method,
            rows,
            columns,
            itemsize,
        )
        preparation_workspace_bytes = batch_count * rows * columns * itemsize
    elif sparse_direct:
        factorization_bytes = rows * columns * itemsize
        preparation_workspace_bytes = rows * columns * itemsize
    elif structured_direct:
        factorization_bytes = structured_factor_bytes
        preparation_workspace_bytes = max(
            (rows + columns) * itemsize,
            structured_factor_bytes,
        )
    else:
        factorization_bytes = 0
        preparation_workspace_bytes = 0
    solve_workspace_bytes_per_rhs = batch_count * (rows + columns) * itemsize
    primal_krylov_bytes = (
        _krylov_storage_bytes(problem, method, policy, itemsize) if iterative else 0
    )
    implicit_krylov_bytes = _implicit_storage_bytes(problem, policy, itemsize)
    krylov_basis_bytes_per_rhs = max(primal_krylov_bytes, implicit_krylov_bytes)
    return LinearCostEstimate(
        provider=backend,
        method=method.name,
        existing_storage_bytes=existing,
        additional_matrix_bytes=(
            structured_dense_bytes
            if structured_direct
            else batch_count * rows * columns * itemsize
            if dense_requires_materialization
            else 0
        ),
        factorization_bytes=factorization_bytes,
        preparation_workspace_bytes=preparation_workspace_bytes,
        solve_workspace_bytes_per_rhs=solve_workspace_bytes_per_rhs,
        krylov_basis_bytes_per_rhs=krylov_basis_bytes_per_rhs,
        operation_class=(
            "structured-direct"
            if structured_direct
            else "direct-factorization"
            if dense_direct or sparse_direct
            else "iterative-matvec"
        ),
        accepted=True,
        reason=reason,
    )


def _krylov_storage_bytes(
    problem: AbstractLinearProblem,
    method: AbstractLinearMethod,
    policy: LinearSolvePolicy,
    itemsize: int,
    /,
) -> int:
    rows = problem.operator.target.size
    columns = problem.operator.source.size
    if isinstance(problem, LeastSquaresProblem) and problem.regularizer is not None:
        rows += problem.regularizer.target.size
    if isinstance(method, FGMRES):
        restart = method.restart
        primal = ((2 * restart + 1) * columns + (restart + 1) * restart) * itemsize
    elif isinstance(method, GMRES):
        restart = method.restart
        primal = ((2 * restart + 1) * columns + (restart + 1) * restart) * itemsize
    elif isinstance(method, GeneralizedLSMR):
        primal = (5 * columns + 2 * rows) * itemsize
    elif isinstance(method, LSMR):
        primal = (5 * columns + 2 * rows) * itemsize
    elif isinstance(method, MINRES):
        primal = 10 * columns * itemsize
    elif isinstance(method, (PCG, ConjugateGradient)):
        primal = 6 * columns * itemsize
    elif isinstance(method, BiCGStab):
        primal = 10 * columns * itemsize
    else:
        primal = 0
    batch_count = prod(problem.operator.batch_shape or (1,))
    if policy.differentiation.mode not in ("mathematical", "rhs-only"):
        return batch_count * primal
    max_steps = policy.tolerance.max_steps or columns
    restart = min(30, max_steps, columns)
    tangent = ((2 * restart + 1) * columns + (restart + 1) * restart) * itemsize
    return batch_count * max(primal, tangent)


def _implicit_storage_bytes(
    problem: AbstractLinearProblem,
    policy: LinearSolvePolicy,
    itemsize: int,
    /,
) -> int:
    if policy.differentiation.mode not in ("mathematical", "rhs-only"):
        return 0
    dimension = problem.operator.source.size
    if isinstance(problem, MinimumNormProblem):
        dimension += problem.operator.target.size
    max_steps = policy.tolerance.max_steps or dimension
    restart = min(30, max_steps, dimension)
    per_problem = ((2 * restart + 1) * dimension + (restart + 1) * restart) * itemsize
    return prod(problem.operator.batch_shape or (1,)) * per_problem


def _require_selected_resources(
    estimate: LinearCostEstimate,
    policy: LinearSolvePolicy,
    /,
) -> None:
    checks = (
        (
            "factorization",
            estimate.factorization_bytes,
            policy.resources.factorization_bytes,
        ),
        (
            "preparation workspace",
            estimate.preparation_workspace_bytes,
            policy.resources.workspace_bytes,
        ),
        (
            "solve workspace per right-hand side",
            estimate.solve_workspace_bytes_per_rhs,
            policy.resources.workspace_bytes,
        ),
        (
            "Krylov basis per right-hand side",
            estimate.krylov_basis_bytes_per_rhs,
            policy.resources.krylov_basis_bytes,
        ),
    )
    for name, required, available in checks:
        if required > available:
            raise ValueError(
                f"Selected {estimate.method} requires {required} {name} bytes, "
                f"exceeding the policy budget {available}."
            )


def _rejected_estimate(entry: str, /) -> LinearCostEstimate:
    method, separator, reason = entry.partition(": ")
    return LinearCostEstimate(
        provider="candidate",
        method=method if separator else "unknown",
        operation_class="rejected",
        accepted=False,
        reason=reason if separator else entry,
    )


def _is_real(operator: AbstractLinearOperator, /) -> bool:
    return all(
        not jnp.issubdtype(spec.dtype, jnp.complexfloating)
        for spec in jax.tree.leaves(operator.source.structure())
        + jax.tree.leaves(operator.target.structure())
    )


def _is_explicit_operator(operator: AbstractLinearOperator, /) -> bool:
    if isinstance(
        operator,
        (
            DenseLinearOperator,
            DiagonalLinearOperator,
            IdentityLinearOperator,
            PermutationLinearOperator,
            TriangularLinearOperator,
            TridiagonalLinearOperator,
            BandedLinearOperator,
            LowRankLinearOperator,
            SymmetricLowRankLinearOperator,
            DiagonalPlusLowRankLinearOperator,
        ),
    ):
        return True
    if isinstance(
        operator,
        (TransposeLinearOperator, AdjointLinearOperator, ScaledLinearOperator),
    ):
        return _is_explicit_operator(operator.operator)
    if isinstance(operator, (SumLinearOperator, ComposedLinearOperator)):
        return _is_explicit_operator(operator.left) and _is_explicit_operator(
            operator.right
        )
    if isinstance(operator, BlockLinearOperator):
        return all(
            _is_explicit_operator(block)
            for row in operator.blocks
            for block in row
            if block is not None
        )
    if isinstance(
        operator,
        (
            BlockDiagonalLinearOperator,
            KroneckerLinearOperator,
            KroneckerSumLinearOperator,
        ),
    ):
        children = (
            operator.blocks
            if isinstance(operator, BlockDiagonalLinearOperator)
            else operator.factors
        )
        return all(_is_explicit_operator(child) for child in children)
    return False


def _cuda_sparse_available() -> bool:
    return any(
        device.platform == "gpu" and "nvidia" in device.device_kind.lower()
        for device in jax.devices()
    )


def _certified_positive_definite(operator: AbstractLinearOperator, /) -> bool:
    return operator.properties.certifies("positive_definite")


def _certified_self_adjoint(operator: AbstractLinearOperator, /) -> bool:
    return operator.properties.certifies("self_adjoint")


def _certified_rank(operator: AbstractLinearOperator, /) -> int | None:
    properties = operator.properties
    if properties.certifies("rank"):
        return properties.rank
    if _certified_positive_definite(operator):
        return operator.source.size
    return None


def _certifies_full_rank(operator: AbstractLinearOperator, /) -> bool:
    return _certified_rank(operator) == min(
        operator.source.size,
        operator.target.size,
    )


def _method_configuration(
    selected_name: str,
    policy: LinearSolvePolicy,
    /,
) -> dict[str, object]:
    method = policy.method
    if isinstance(method, AutoLinearMethod):
        defaults: dict[str, AbstractLinearMethod] = {
            GMRES().name: GMRES(),
            FGMRES().name: FGMRES(),
            LSMR().name: LSMR(),
            GeneralizedLSMR().name: GeneralizedLSMR(),
        }
        method = defaults.get(selected_name, method)
    configuration: dict[str, object] = {"name": selected_name}
    if isinstance(method, (GMRES, FGMRES)):
        configuration["restart"] = method.restart
        configuration["stagnation_iterations"] = method.stagnation_iterations
    elif isinstance(method, (LSMR, GeneralizedLSMR)):
        configuration["condition_limit"] = method.condition_limit
        configuration["damping"] = method.damping
    elif isinstance(method, SparseDirect):
        configuration["reorder"] = method.reorder
    return configuration


__all__ = ["LinearBackend", "LinearSolvePlan", "plan"]
