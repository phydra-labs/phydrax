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
    estimate_operator_action_cost,
    IdentityLinearOperator,
    ScaledLinearOperator,
    SumLinearOperator,
    TransposeLinearOperator,
)
from ._policies import (
    AbstractLinearMethod,
    AutoLinearMethod,
    BiCGStab,
    BlockCG,
    BlockGMRES,
    ConjugateGradient,
    DenseCholesky,
    DenseLU,
    DenseQR,
    DenseSVD,
    FGMRES,
    GeneralizedLSMR,
    GMRES,
    LinearSolvePolicy,
    LSMR,
    MINRES,
    PCG,
    ProjectedPCG,
    SparseCholesky,
    SparseLDLT,
    SparseLU,
    SparseQR,
    StructuredDirect,
)
from ._preconditioning import JacobiPreconditionerBuilder, PreconditionerPlan
from ._problems import (
    _problem_structure,
    AbstractLinearProblem,
    LeastSquaresProblem,
    LinearSystem,
    MinimumNormProblem,
)
from ._properties import LinearCapabilityError
from ._spaces import (
    _coordinate_dtype,
    _has_diagonal_pairing,
    _has_euclidean_pairing,
    RHSLayout,
)
from ._sparse_contract import AbstractSparseLinearOperator
from ._sparse_providers import sparse_provider_availability, SparseProviderName
from ._structured_operators import (
    _is_structured_exact,
    BandedLinearOperator,
    BlockDiagonalLinearOperator,
    DiagonalPlusLowRankLinearOperator,
    KroneckerLinearOperator,
    KroneckerSumLinearOperator,
    LocalBlockDiagonalLinearOperator,
    LowRankLinearOperator,
    PermutationLinearOperator,
    SymmetricLowRankLinearOperator,
    TriangularLinearOperator,
    TridiagonalLinearOperator,
)
from ._transform_operators import TransformDiagonalLinearOperator


LinearBackend: TypeAlias = Literal[
    "jax-structured",
    "jax-dense",
    "jax-sparse",
    "host-sparse",
    "spineax-cudss",
    "native-krylov",
    "native-block-krylov",
    "matfree",
    "lineax",
]


class LinearSolvePlan(StrictModule):
    """Immutable symbolic selection; all numerical state belongs to preparation."""

    policy: LinearSolvePolicy
    candidates: tuple[LinearCostEstimate, ...]
    preconditioner_plan: PreconditionerPlan | None
    rhs_layout: RHSLayout | None
    problem_id: str = eqx.field(static=True)
    problem_kind: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    problem_signature: str = eqx.field(static=True)
    backend: LinearBackend = eqx.field(static=True)
    method: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    rejected: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    recycling_capacity: int = eqx.field(static=True)
    recycling_state_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: AbstractLinearProblem,
        policy: LinearSolvePolicy,
        rhs_layout: RHSLayout | None = None,
        backend: LinearBackend,
        preconditioner_plan: PreconditionerPlan | None,
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
            "spineax-cudss",
            "native-krylov",
            "native-block-krylov",
            "matfree",
            "lineax",
        ):
            raise ValueError("Unknown linear backend.")
        values = (str(method), str(reason))
        if any(not value for value in values):
            raise ValueError("Plan method and reason must be non-empty.")
        if (policy.preconditioning is None) != (preconditioner_plan is None):
            raise ValueError(
                "Solve and preconditioner plans must agree on preconditioning."
            )
        if (
            preconditioner_plan is not None
            and preconditioner_plan.space_id != problem.operator.source.space_id
        ):
            raise ValueError(
                "Preconditioner plan space must match the problem source space."
            )
        if rhs_layout is not None and not isinstance(rhs_layout, RHSLayout):
            raise TypeError("rhs_layout must be an RHSLayout or None.")
        self.policy = policy
        self.preconditioner_plan = preconditioner_plan
        self.candidates = candidates
        self.rhs_layout = rhs_layout
        self.problem_id = problem.problem_id
        self.problem_kind = problem.kind
        self.operator_id = problem.operator.operator_id
        self.problem_signature = _problem_structure(problem)
        self.backend = backend
        self.method, self.reason = values
        self.rejected = tuple(str(value) for value in rejected)
        self.recycling_capacity = (
            0 if policy.recycling is None else policy.recycling.capacity
        )
        self.recycling_state_bytes = _recycling_state_bytes(problem, policy)
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
                "rhs_layout": None if rhs_layout is None else rhs_layout.layout_id,
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
                **(
                    {}
                    if policy.precision is None
                    else {
                        "precision": {
                            "operator_dtype": policy.precision.operator_dtype,
                            "factorization_dtype": (policy.precision.factorization_dtype),
                            "preconditioner_dtype": (
                                policy.precision.preconditioner_dtype
                            ),
                            "krylov_dtype": policy.precision.krylov_dtype,
                            "residual_dtype": policy.precision.residual_dtype,
                            "accumulation_dtype": (policy.precision.accumulation_dtype),
                            "maximum_refinement_steps": (
                                policy.precision.maximum_refinement_steps
                            ),
                            "condition_limit": policy.precision.condition_limit,
                        }
                    }
                ),
                "preconditioning": (
                    None if preconditioner_plan is None else preconditioner_plan.plan_id
                ),
                "recycling": (
                    None
                    if policy.recycling is None
                    else {
                        "capacity": policy.recycling.capacity,
                        "extraction": policy.recycling.extraction,
                        "refresh": policy.recycling.refresh,
                    }
                ),
                "differentiation": policy.differentiation.mode,
                "failure": policy.failure.mode,
                "require_device_binding": policy.require_device_binding,
                "materialization": {
                    "max_entries": policy.materialization.max_entries,
                    "max_bytes": policy.materialization.max_bytes,
                },
                "resources": {
                    "factorization_bytes": policy.resources.factorization_bytes,
                    "workspace_bytes": policy.resources.workspace_bytes,
                    "krylov_basis_bytes": policy.resources.krylov_basis_bytes,
                    "preconditioner_bytes": policy.resources.preconditioner_bytes,
                    "recycling_state_bytes": policy.resources.recycling_state_bytes,
                },
            }
        )


def plan(
    problem: AbstractLinearProblem,
    policy: LinearSolvePolicy | None = None,
    /,
    *,
    rhs_layout: RHSLayout | None = None,
) -> LinearSolvePlan:
    """Select the first feasible candidate in deterministic capability order."""
    if not isinstance(problem, AbstractLinearProblem):
        raise TypeError("problem must be an AbstractLinearProblem.")
    policy_ = LinearSolvePolicy() if policy is None else policy
    if not isinstance(policy_, LinearSolvePolicy):
        raise TypeError("policy must be a LinearSolvePolicy.")
    if rhs_layout is not None and not isinstance(rhs_layout, RHSLayout):
        raise TypeError("rhs_layout must be an RHSLayout or None.")
    requested = policy_.method
    selected, reason, rejected = (
        _auto_method(problem, policy_)
        if isinstance(requested, AutoLinearMethod)
        else (requested, "explicit policy", ())
    )
    if policy_.recycling is not None:
        if not isinstance(selected, (GMRES, FGMRES)):
            raise ValueError("RecyclingPolicy requires GMRES or FGMRES.")
        if policy_.differentiation.mode == "algorithmic":
            raise ValueError(
                "Algorithmic differentiation through GCRO-DR is unsupported."
            )
        if policy_.preconditioning is not None:
            raise ValueError(
                "GCRO-DR recycling with a preconditioner is not yet supported."
            )
    backend = _validate_method(problem, selected, policy_, rhs_layout)
    _validate_precision_policy(problem, selected, backend, policy_)
    if policy_.require_device_binding and backend in ("host-sparse", "lineax"):
        raise ValueError(
            f"Selected backend {backend!r} cannot bind numerical state on device."
        )
    preconditioner_plan = _make_preconditioner_plan(
        problem,
        selected,
        policy_,
    )
    selected_estimate = _selected_estimate(
        problem,
        selected,
        backend,
        policy_,
        preconditioner_plan,
        rhs_layout,
        reason,
    )
    _require_selected_resources(
        selected_estimate,
        policy_,
        1 if rhs_layout is None else rhs_layout.size,
    )
    estimates = tuple(_rejected_estimate(entry) for entry in rejected) + (
        selected_estimate,
    )
    return LinearSolvePlan(
        problem=problem,
        policy=policy_,
        backend=backend,
        preconditioner_plan=preconditioner_plan,
        rhs_layout=rhs_layout,
        method=selected.name,
        reason=reason,
        rejected=rejected,
        candidates=estimates,
    )


def _validate_precision_policy(
    problem: AbstractLinearProblem,
    method: AbstractLinearMethod,
    backend: LinearBackend,
    policy: LinearSolvePolicy,
    /,
) -> None:
    precision = policy.precision
    if precision is None:
        return

    rejection = "MixedPrecisionPolicy is capability-rejected"
    if isinstance(method, SparseLDLT):
        coordinate_dtype = _coordinate_dtype(problem.operator.source)
        requested_operator = (
            coordinate_dtype
            if precision.operator_dtype is None
            else jnp.dtype(precision.operator_dtype)
        )
        factorization_dtype = (
            coordinate_dtype
            if precision.factorization_dtype is None
            else jnp.dtype(precision.factorization_dtype)
        )
        if requested_operator != coordinate_dtype:
            raise LinearCapabilityError(
                f"{rejection}: operator_dtype must match stored sparse coordinates."
            )
        if factorization_dtype != coordinate_dtype:
            raise LinearCapabilityError(
                f"{rejection}: Spineax cuDSS currently requires factorization_dtype "
                "to match stored sparse coordinates."
            )
        if (
            precision.preconditioner_dtype is not None
            or precision.krylov_dtype is not None
        ):
            raise LinearCapabilityError(
                f"{rejection}: sparse LDLT has no preconditioner or Krylov storage."
            )
        return
    iterative_methods = (
        PCG,
        ProjectedPCG,
        MINRES,
        ConjugateGradient,
        BlockCG,
        GMRES,
        FGMRES,
        BiCGStab,
        BlockGMRES,
    )
    if isinstance(method, iterative_methods):
        if backend not in ("native-krylov", "native-block-krylov"):
            raise LinearCapabilityError(
                f"{rejection}: iterative mixed precision requires a native Krylov provider."
            )
        coordinate_dtype = _coordinate_dtype(problem.operator.source)
        requested_operator = (
            coordinate_dtype
            if precision.operator_dtype is None
            else jnp.dtype(precision.operator_dtype)
        )
        residual_dtype = (
            coordinate_dtype
            if precision.residual_dtype is None
            else jnp.dtype(precision.residual_dtype)
        )
        accumulation_dtype = (
            coordinate_dtype
            if precision.accumulation_dtype is None
            else jnp.dtype(precision.accumulation_dtype)
        )
        if requested_operator != coordinate_dtype:
            raise LinearCapabilityError(
                f"{rejection}: operator_dtype must match stored coordinates."
            )
        if residual_dtype != coordinate_dtype or accumulation_dtype != coordinate_dtype:
            raise LinearCapabilityError(
                f"{rejection}: native Krylov residual and accumulation must remain "
                "in the stored coordinate precision."
            )
        if precision.factorization_dtype is not None:
            raise LinearCapabilityError(
                f"{rejection}: iterative methods have no factorization stage."
            )
        if precision.preconditioner_dtype is None and precision.krylov_dtype is None:
            raise LinearCapabilityError(
                f"{rejection}: iterative precision requires a preconditioner or "
                "Krylov basis dtype."
            )
        if precision.krylov_dtype is not None:
            if not isinstance(method, (GMRES, FGMRES)):
                raise LinearCapabilityError(
                    f"{rejection}: compressed basis storage currently supports "
                    "GMRES/FGMRES only."
                )
            krylov_dtype = jnp.dtype(precision.krylov_dtype)
            same_kind = jnp.issubdtype(
                coordinate_dtype, jnp.complexfloating
            ) == jnp.issubdtype(krylov_dtype, jnp.complexfloating)
            if not same_kind or krylov_dtype.itemsize > coordinate_dtype.itemsize:
                raise LinearCapabilityError(
                    f"{rejection}: Krylov dtype must have the coordinate kind and "
                    "no greater precision."
                )
        if precision.preconditioner_dtype is not None:
            if policy.preconditioning is None:
                raise LinearCapabilityError(
                    f"{rejection}: preconditioner_dtype requires preconditioning."
                )
            preconditioner_dtype = jnp.dtype(precision.preconditioner_dtype)
            same_kind = jnp.issubdtype(
                coordinate_dtype, jnp.complexfloating
            ) == jnp.issubdtype(preconditioner_dtype, jnp.complexfloating)
            if not same_kind or preconditioner_dtype.itemsize > coordinate_dtype.itemsize:
                raise LinearCapabilityError(
                    f"{rejection}: preconditioner dtype must have the coordinate kind "
                    "and no greater precision."
                )
        if precision.maximum_refinement_steps or precision.condition_limit is not None:
            raise LinearCapabilityError(
                f"{rejection}: iterative refinement controls apply only to DenseLU."
            )
        return
    if backend != "jax-dense" or not isinstance(method, DenseLU):
        raise LinearCapabilityError(
            f"{rejection}: only the jax-dense DenseLU provider is supported."
        )
    if not isinstance(problem, LinearSystem) or not isinstance(
        problem.operator,
        DenseLinearOperator,
    ):
        raise LinearCapabilityError(
            f"{rejection}: mixed-precision DenseLU requires explicit dense square "
            "storage and never materializes another operator."
        )
    if not (
        _has_euclidean_pairing(problem.operator.source)
        and _has_euclidean_pairing(problem.operator.target)
    ):
        raise LinearCapabilityError(
            f"{rejection}: mixed-precision refinement currently requires "
            "Euclidean source and target pairings."
        )
    if precision.preconditioner_dtype is not None:
        raise LinearCapabilityError(
            f"{rejection}: jax-dense DenseLU has no preconditioner arithmetic."
        )
    if precision.krylov_dtype is not None:
        raise LinearCapabilityError(
            f"{rejection}: jax-dense DenseLU has no Krylov arithmetic."
        )

    operator_dtype = jnp.dtype(problem.operator.matrix.dtype)
    requested_operator = (
        operator_dtype
        if precision.operator_dtype is None
        else jnp.dtype(precision.operator_dtype)
    )
    residual_dtype = (
        operator_dtype
        if precision.residual_dtype is None
        else jnp.dtype(precision.residual_dtype)
    )
    accumulation_dtype = (
        operator_dtype
        if precision.accumulation_dtype is None
        else jnp.dtype(precision.accumulation_dtype)
    )
    factorization_dtype = (
        operator_dtype
        if precision.factorization_dtype is None
        else jnp.dtype(precision.factorization_dtype)
    )
    if requested_operator != operator_dtype:
        raise LinearCapabilityError(
            f"{rejection}: operator_dtype={requested_operator.name!r} does not "
            f"match stored coordinates {operator_dtype.name!r}."
        )
    if residual_dtype != operator_dtype:
        raise LinearCapabilityError(
            f"{rejection}: jax-dense certification requires residual_dtype to "
            "match the stored operator precision."
        )
    if accumulation_dtype != operator_dtype:
        raise LinearCapabilityError(
            f"{rejection}: jax-dense refinement requires accumulation_dtype to "
            "match the stored operator precision."
        )
    factorization_supported = (
        jnp.dtype(jnp.float32),
        jnp.dtype(jnp.float64),
        jnp.dtype(jnp.complex64),
        jnp.dtype(jnp.complex128),
    )
    if factorization_dtype not in factorization_supported:
        raise LinearCapabilityError(
            f"{rejection}: jax-dense LU does not support "
            f"factorization_dtype={factorization_dtype.name!r}."
        )
    same_scalar_kind = jnp.issubdtype(
        operator_dtype, jnp.complexfloating
    ) == jnp.issubdtype(factorization_dtype, jnp.complexfloating)
    if not same_scalar_kind or factorization_dtype.itemsize > operator_dtype.itemsize:
        raise LinearCapabilityError(
            f"{rejection}: factorization_dtype must have the stored operator's "
            "real/complex kind and no greater precision."
        )
    lower_factorization = factorization_dtype.itemsize < operator_dtype.itemsize
    if (
        precision.maximum_refinement_steps > 0 or precision.condition_limit is not None
    ) and not lower_factorization:
        raise LinearCapabilityError(
            f"{rejection}: refinement and condition screening require a lower "
            "factorization precision."
        )


def _make_preconditioner_plan(
    problem: AbstractLinearProblem,
    method: AbstractLinearMethod,
    policy: LinearSolvePolicy,
    /,
) -> PreconditionerPlan | None:
    preconditioning = policy.preconditioning
    if preconditioning is None:
        return None
    preconditioner_dtype = (
        None if policy.precision is None else policy.precision.preconditioner_dtype
    )
    if preconditioner_dtype is not None and not isinstance(
        preconditioning.builder,
        JacobiPreconditionerBuilder,
    ):
        raise LinearCapabilityError(
            "Lower-precision preconditioning currently supports Jacobi builders only."
        )
    if isinstance(method, (PCG, ProjectedPCG, MINRES, ConjugateGradient, BlockCG)):
        required_side: Literal["left", "right"] = "left"
    elif isinstance(method, (GMRES, FGMRES, BiCGStab, BlockGMRES)):
        required_side = "right"
    else:
        raise ValueError(f"{method.name} does not accept preconditioning.")
    if preconditioning.side not in ("auto", required_side):
        raise ValueError(
            f"{method.name} requires {required_side} preconditioning; "
            f"got {preconditioning.side!r}."
        )
    return PreconditionerPlan(
        preconditioning,
        problem.operator,
        side=required_side,
        materialization=policy.materialization,
        compute_dtype=preconditioner_dtype,
    )


def _preconditioner_properties(
    problem: AbstractLinearProblem,
    policy: LinearSolvePolicy,
    /,
):
    if policy.preconditioning is None:
        return None
    return policy.preconditioning.properties_for(problem.operator)


def _auto_method(
    problem: AbstractLinearProblem,
    policy: LinearSolvePolicy,
    /,
) -> tuple[AbstractLinearMethod, str, tuple[str, ...]]:
    operator = problem.operator
    rejected: list[str] = []
    projected_rejection = _projected_pcg_rejection(problem)
    if projected_rejection is None:
        preconditioner_properties = _preconditioner_properties(problem, policy)
        if preconditioner_properties is None or (
            preconditioner_properties.certifies("positive_definite")
            and preconditioner_properties.certifies("self_adjoint")
            and preconditioner_properties.certifies("linear")
            and preconditioner_properties.certifies("stationary")
        ):
            return (
                ProjectedPCG(),
                "complete certified kernel with positive semidefinite quotient",
                (),
            )
        rejected.append(
            "projected-pcg: requires a fixed, linear, self-adjoint, "
            "positive-definite preconditioner"
        )
    elif (
        isinstance(problem, LinearSystem)
        and problem.nullspace_policy is not None
        and problem.nullspace_policy.certificate is not None
    ):
        rejected.append(f"projected-pcg: {projected_rejection}")
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
            and policy.preconditioning is None
            and policy.rank.relative_cutoff is None
        ):
            return SparseQR(), "canonical CSR with native CUDA sparse QR", ()
        rejected.append(
            "sparse-qr/jax-cuda: execution requires CUDA, no preconditioning, "
            "and no numerical rank cutoff"
        )
    if isinstance(problem, LinearSystem):
        dense_method: AbstractLinearMethod
        if _certified_positive_definite(operator) and _has_diagonal_pairing(
            operator.source
        ):
            dense_method = DenseCholesky()
        else:
            dense_method = DenseLU()
        if explicit and policy.preconditioning is None:
            fits, explanation = _dense_candidate_fits(problem, dense_method, policy)
            if fits:
                return dense_method, explanation, ()
            rejected.append(f"{dense_method.name}: {explanation}")
        elif explicit:
            rejected.append(
                f"{dense_method.name}: dense direct execution does not accept "
                "preconditioning"
            )
        properties = _preconditioner_properties(problem, policy)
        preconditioner_is_positive = properties is None or properties.certifies(
            "positive_definite"
        )
        if _certified_positive_definite(operator) and preconditioner_is_positive:
            return PCG(), "positive-definite Krylov fallback", tuple(rejected)
        if _certified_self_adjoint(operator) and preconditioner_is_positive:
            return MINRES(), "self-adjoint indefinite Krylov fallback", tuple(rejected)
        if policy.preconditioning is not None:
            return (
                FGMRES(),
                "general flexible-preconditioned Krylov fallback",
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
        if explicit and policy.preconditioning is None:
            fits, explanation = _dense_candidate_fits(problem, DenseSVD(), policy)
            if fits:
                return DenseSVD(), explanation, ()
            rejected.append(f"dense-svd: {explanation}")
        elif explicit:
            rejected.append(
                "dense-svd: dense direct execution does not accept preconditioning"
            )
        if _matfree_lsmr_eligible(problem, policy):
            return LSMR(), "real-Euclidean Matfree LSMR envelope", tuple(rejected)
        return (
            GeneralizedLSMR(),
            "pairing-aware generalized least-squares Krylov fallback",
            tuple(rejected),
        )
    raise TypeError(f"Unsupported problem type {type(problem).__name__}.")


def _projected_pcg_rejection(problem: AbstractLinearProblem, /) -> str | None:
    if not isinstance(problem, LinearSystem):
        return "requires a LinearSystem"
    operator = problem.operator
    if not _certified_self_adjoint(operator):
        return "requires certified self-adjoint structure"
    if not operator.properties.certifies("positive_semidefinite"):
        return "requires certified positive semidefiniteness"
    nullspace = problem.nullspace_policy
    if nullspace is None or nullspace.right is None:
        return "requires an explicit right-nullspace policy"
    certificate = nullspace.certificate
    if certificate is None:
        return "requires a KernelCertificate"
    if not certificate.complete:
        return "requires a complete kernel/nullity certificate"
    if certificate.right.capacity == 0:
        return "requires a nonempty certified kernel"
    if certificate.right.subspace_id != nullspace.right.subspace_id:
        return "received a kernel certificate for a different right subspace"
    if certificate.operator_id != operator.operator_id:
        return "received a kernel certificate for a different operator structure"
    if not certificate.matches(operator):
        return "received a stale numerical kernel certificate"
    return None


def _validate_method(
    problem: AbstractLinearProblem,
    method: AbstractLinearMethod,
    policy: LinearSolvePolicy,
    rhs_layout: RHSLayout | None,
    /,
) -> LinearBackend:
    operator = problem.operator
    preconditioner = policy.preconditioning
    preconditioner_properties = _preconditioner_properties(problem, policy)
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
    if isinstance(method, (SparseQR, SparseLU, SparseCholesky, SparseLDLT)):
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
        if isinstance(method, SparseCholesky) and not operator.properties.certifies(
            "positive_definite"
        ):
            raise ValueError(
                "SparseCholesky requires certified positive-definite structure."
            )
        if isinstance(method, SparseLDLT):
            if not operator.properties.certifies("self_adjoint"):
                raise ValueError("SparseLDLT requires certified self-adjoint structure.")
            if operator.sparse_storage().index_width != 32:
                raise ValueError("Spineax cuDSS execution requires 32-bit CSR indices.")
            if policy.differentiation.mode == "algorithmic":
                raise ValueError(
                    "SparseLDLT exposes mathematical differentiation, not an "
                    "algorithmic factorization derivative."
                )
            availability = sparse_provider_availability(method.provider)
            if not availability.available:
                raise ValueError(availability.reason)
            return "spineax-cudss"
        if isinstance(method, SparseQR) and method.provider == "jax-cuda":
            if not _cuda_sparse_available():
                raise ValueError(
                    "SparseQR(provider='jax-cuda') requires a JAX CUDA device."
                )
            if policy.differentiation.mode == "algorithmic":
                raise ValueError(
                    "SparseQR exposes mathematical differentiation, not an "
                    "algorithmic QR derivative."
                )
            return "jax-sparse"
        if policy.differentiation.mode != "none":
            raise ValueError(
                "Host sparse direct providers are non-JIT and require "
                "DifferentiationPolicy('none')."
            )
        if isinstance(method, SparseLU):
            provider: SparseProviderName = (
                "scipy-superlu" if method.provider == "auto" else method.provider
            )
        else:
            provider = method.provider
        availability = sparse_provider_availability(provider)
        if not availability.available:
            raise ValueError(availability.reason)
        return "host-sparse"
    if operator.batch_shape:
        raise ValueError("Iterative providers require explicit batched execution policy.")
    square_iterative = isinstance(
        method,
        (
            PCG,
            ProjectedPCG,
            MINRES,
            FGMRES,
            ConjugateGradient,
            GMRES,
            BiCGStab,
            BlockCG,
            BlockGMRES,
        ),
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
    if isinstance(method, (BlockCG, BlockGMRES)):
        if rhs_layout is None or rhs_layout.size <= 1:
            raise ValueError(
                f"{method.name} requires a planned layout with multiple right-hand sides."
            )
        if not isinstance(problem, LinearSystem):
            raise ValueError(f"{method.name} requires a LinearSystem.")
        if policy.differentiation.mode == "algorithmic":
            raise ValueError(
                "Block Krylov rank transitions do not expose algorithmic differentiation."
            )
        if isinstance(method, BlockCG):
            if not (
                _certified_self_adjoint(operator)
                and _certified_positive_definite(operator)
            ):
                raise ValueError(
                    "BlockCG requires certified self-adjoint positive-definite structure."
                )
            if preconditioner_properties is not None and not (
                preconditioner_properties.certifies("positive_definite")
                and preconditioner_properties.certifies("self_adjoint")
                and preconditioner_properties.certifies("linear")
                and preconditioner_properties.certifies("stationary")
            ):
                raise ValueError(
                    "BlockCG requires a fixed, linear, self-adjoint, "
                    "positive-definite left preconditioner."
                )
        elif preconditioner_properties is not None and not (
            preconditioner_properties.certifies("linear")
            and preconditioner_properties.certifies("stationary")
        ):
            raise ValueError("BlockGMRES requires fixed linear right preconditioning.")
        return "native-block-krylov"
    if isinstance(method, ProjectedPCG):
        rejection = _projected_pcg_rejection(problem)
        if rejection is not None:
            raise ValueError(f"ProjectedPCG {rejection}.")
        if preconditioner_properties is not None and not (
            preconditioner_properties.certifies("positive_definite")
            and preconditioner_properties.certifies("self_adjoint")
            and preconditioner_properties.certifies("linear")
            and preconditioner_properties.certifies("stationary")
        ):
            raise ValueError(
                "ProjectedPCG requires a fixed, linear, self-adjoint, "
                "positive-definite preconditioner."
            )
        return "native-krylov"
    if isinstance(method, PCG):
        if not isinstance(problem, LinearSystem):
            raise ValueError("PCG requires a LinearSystem.")
        if not _certified_positive_definite(operator):
            raise ValueError("PCG requires certified positive definiteness.")
        if preconditioner_properties is not None and not (
            preconditioner_properties.certifies("positive_definite")
            and preconditioner_properties.certifies("self_adjoint")
            and preconditioner_properties.certifies("linear")
            and preconditioner_properties.certifies("stationary")
        ):
            raise ValueError(
                "PCG requires a fixed, linear, self-adjoint, "
                "positive-definite preconditioner."
            )
        return "native-krylov"
    if isinstance(method, MINRES):
        if not isinstance(problem, LinearSystem):
            raise ValueError("MINRES requires a LinearSystem.")
        if not _certified_self_adjoint(operator):
            raise ValueError("MINRES requires certified self-adjoint structure.")
        if preconditioner_properties is not None and not (
            preconditioner_properties.certifies("positive_definite")
            and preconditioner_properties.certifies("self_adjoint")
            and preconditioner_properties.certifies("linear")
            and preconditioner_properties.certifies("stationary")
        ):
            raise ValueError(
                "MINRES requires a fixed, linear, self-adjoint, "
                "positive-definite preconditioner."
            )
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
        if preconditioner_properties is not None and not (
            preconditioner_properties.certifies("positive_definite")
            and preconditioner_properties.certifies("linear")
            and preconditioner_properties.certifies("stationary")
        ):
            raise ValueError(
                "CG requires a fixed, linear, positive-definite preconditioner."
            )
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
        if preconditioner_properties is not None and not (
            preconditioner_properties.certifies("linear")
            and preconditioner_properties.certifies("stationary")
        ):
            raise ValueError(
                "GMRES requires fixed linear preconditioning; use FGMRES for "
                "variable or nonlinear actions."
            )
        return "native-krylov"
    if isinstance(method, BiCGStab):
        if not isinstance(problem, LinearSystem):
            raise ValueError("bicgstab requires a LinearSystem.")
        if not _has_diagonal_pairing(operator.source):
            raise ValueError(
                "Lineax methods require a Euclidean or diagonal source pairing."
            )
        if preconditioner_properties is not None and not (
            preconditioner_properties.certifies("linear")
            and preconditioner_properties.certifies("stationary")
        ):
            raise ValueError(
                "BiCGStab requires fixed linear preconditioning; use FGMRES for "
                "variable or nonlinear actions."
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
    if policy.preconditioning is not None or policy.differentiation.mode == "algorithmic":
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
    if isinstance(operator, KroneckerSumLinearOperator):
        return sum(
            0
            if isinstance(factor, DenseLinearOperator)
            else factor.source.size * factor.target.size
            for factor in operator.factors
        )
    if isinstance(operator, LocalBlockDiagonalLinearOperator):
        return 0
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
    if isinstance(operator, TransformDiagonalLinearOperator):
        return operator.source.size + 1
    if isinstance(operator, DiagonalPlusLowRankLinearOperator):
        dimension = operator.source.size
        rank = operator.left_factor.shape[1]
        core_lu = rank * rank + rank
        woodbury = core_lu + dimension * rank + dimension
        if operator.nonsingular_diagonal:
            return woodbury
        dense_lu = dimension * dimension + dimension
        return dense_lu + woodbury
    if isinstance(operator, LocalBlockDiagonalLinearOperator):
        return operator.blocks.size + operator.num_blocks * operator.input_block_size
    if isinstance(operator, BlockDiagonalLinearOperator):
        return sum(_structured_factorization_entries(block) for block in operator.blocks)
    if isinstance(operator, KroneckerLinearOperator):
        return sum(
            _structured_factorization_entries(factor) for factor in operator.factors
        )
    if isinstance(operator, KroneckerSumLinearOperator):
        return (
            sum(
                factor.source.size * factor.source.size + 2 * factor.source.size
                for factor in operator.factors
            )
            + 2 * operator.source.size
            + 1
        )
    return 0


def _structured_factorization_bytes(
    operator: AbstractLinearOperator,
    /,
) -> int:
    itemsize = _coordinate_dtype(operator.source).itemsize
    if isinstance(operator, TransformDiagonalLinearOperator):
        return operator.source.size * itemsize + jnp.dtype(bool).itemsize
    if isinstance(operator, KroneckerSumLinearOperator):
        real_itemsize = (
            max(1, itemsize // 2)
            if jnp.issubdtype(
                _coordinate_dtype(operator.source),
                jnp.complexfloating,
            )
            else itemsize
        )
        factor_dimensions = tuple(factor.source.size for factor in operator.factors)
        return (
            sum(size * size * itemsize for size in factor_dimensions)
            + 2 * sum(factor_dimensions) * real_itemsize
            + operator.source.size * real_itemsize
            + (operator.source.size + 1) * jnp.dtype(bool).itemsize
        )
    if isinstance(operator, LocalBlockDiagonalLinearOperator):
        real_itemsize = jnp.empty((), dtype=operator.blocks.dtype).real.dtype.itemsize
        return (
            operator.blocks.size * itemsize
            + operator.num_blocks
            * operator.input_block_size
            * jnp.dtype(jnp.int32).itemsize
            + operator.num_blocks * operator.input_block_size * real_itemsize
            + operator.num_blocks * jnp.dtype(bool).itemsize
        )
    return _structured_factorization_entries(operator) * itemsize


def _structured_solve_workspace_bytes(
    operator: AbstractLinearOperator,
    /,
) -> int:
    itemsize = _coordinate_dtype(operator.source).itemsize
    if isinstance(operator, TransformDiagonalLinearOperator):
        return 3 * operator.source.size * itemsize
    if isinstance(operator, KroneckerSumLinearOperator):
        return 4 * operator.source.size * itemsize
    return (operator.source.size + operator.target.size) * itemsize


def _structured_candidate_fits(
    problem: AbstractLinearProblem,
    policy: LinearSolvePolicy,
    /,
) -> tuple[bool, str]:
    operator = problem.operator
    if policy.preconditioning is not None:
        return False, "structured direct execution does not accept preconditioners"
    if policy.rank.relative_cutoff is not None:
        return False, "structured direct execution cannot enforce a rank cutoff"
    if policy.rank.require_full_rank and not _certifies_full_rank(operator):
        return False, "full-rank execution lacks a full-rank certificate"
    materialized_entries = _structured_dense_entries(operator)
    itemsize = _coordinate_dtype(operator.source).itemsize
    materialized_bytes = materialized_entries * itemsize
    factor_bytes = _structured_factorization_bytes(operator)
    if materialized_entries > policy.materialization.max_entries:
        return False, (
            f"fallback materialization requires {materialized_entries} entries"
        )
    if materialized_bytes > policy.materialization.max_bytes:
        return False, f"fallback materialization requires {materialized_bytes} bytes"
    if factor_bytes > policy.resources.factorization_bytes:
        return False, f"factorization estimate {factor_bytes} exceeds budget"
    workspace = max(
        _structured_solve_workspace_bytes(operator),
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
    factorization_itemsize = (
        itemsize
        if not isinstance(method, DenseLU)
        or policy.precision is None
        or policy.precision.factorization_dtype is None
        else jnp.dtype(policy.precision.factorization_dtype).itemsize
    )
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
    factor = batch_count * _factorization_bytes(
        method,
        rows,
        columns,
        factorization_itemsize,
    )
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


def _selected_estimate(
    problem: AbstractLinearProblem,
    method: AbstractLinearMethod,
    backend: LinearBackend,
    policy: LinearSolvePolicy,
    preconditioner_plan: PreconditionerPlan | None,
    rhs_layout: RHSLayout | None,
    reason: str,
    /,
) -> LinearCostEstimate:
    rows, columns = problem.operator.target.size, problem.operator.source.size
    if isinstance(problem, LeastSquaresProblem) and problem.regularizer is not None:
        rows += problem.regularizer.target.size
    itemsize = _coordinate_dtype(problem.operator.source).itemsize
    factorization_itemsize = (
        itemsize
        if policy.precision is None or policy.precision.factorization_dtype is None
        else jnp.dtype(policy.precision.factorization_dtype).itemsize
    )
    operator_action_cost = estimate_operator_action_cost(problem.operator)
    existing = operator_action_cost.storage_bytes
    batch_count = prod(problem.operator.batch_shape or (1,))
    dense_direct = backend == "jax-dense"
    regularizer_action_cost = None
    if isinstance(problem, LeastSquaresProblem) and problem.regularizer is not None:
        regularizer_action_cost = estimate_operator_action_cost(problem.regularizer)
        existing += regularizer_action_cost.storage_bytes
    sparse_direct = backend in ("jax-sparse", "host-sparse", "spineax-cudss")
    structured_direct = backend == "jax-structured"
    dense_requires_materialization = dense_direct and _requires_dense_materialization(
        problem
    )
    structured_dense_bytes = (
        _structured_dense_entries(problem.operator) * itemsize if structured_direct else 0
    )
    structured_factor_bytes = (
        _structured_factorization_bytes(problem.operator) if structured_direct else 0
    )
    iterative = backend in (
        "lineax",
        "native-krylov",
        "native-block-krylov",
        "matfree",
    )
    if dense_direct:
        factorization_bytes = batch_count * _factorization_bytes(
            method,
            rows,
            columns,
            factorization_itemsize,
        )
        preparation_workspace_bytes = batch_count * rows * columns * itemsize
    elif sparse_direct:
        factorization_bytes = rows * columns * itemsize
        preparation_workspace_bytes = rows * columns * itemsize
    elif structured_direct:
        factorization_bytes = structured_factor_bytes
        preparation_workspace_bytes = max(
            _structured_solve_workspace_bytes(problem.operator),
            structured_factor_bytes,
        )
    else:
        factorization_bytes = 0
        preparation_workspace_bytes = 0
    if (
        dense_direct
        and policy.precision is not None
        and policy.precision.maximum_refinement_steps > 0
    ):
        solve_workspace_bytes_per_rhs = batch_count * (
            2 * (rows + columns) * itemsize + columns * factorization_itemsize
        )
    else:
        solve_workspace_bytes_per_rhs = (
            _structured_solve_workspace_bytes(problem.operator)
            if structured_direct
            else batch_count * (rows + columns) * itemsize
        )
    rhs_width = 1 if rhs_layout is None else rhs_layout.size
    primal_krylov_bytes = (
        _krylov_storage_bytes(problem, method, policy, itemsize, rhs_width)
        if iterative
        else 0
    )
    implicit_krylov_bytes = _implicit_storage_bytes(problem, policy, itemsize)
    krylov_basis_bytes_per_rhs = max(primal_krylov_bytes, implicit_krylov_bytes)
    operator_apply_workspace_bytes_per_rhs = (
        max(
            operator_action_cost.apply_workspace_bytes_per_rhs,
            0
            if regularizer_action_cost is None
            else regularizer_action_cost.apply_workspace_bytes_per_rhs,
        )
        if iterative
        else 0
    )
    preconditioner_cost = (
        None if preconditioner_plan is None else preconditioner_plan.cost
    )
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
        operator_apply_workspace_bytes_per_rhs=(operator_apply_workspace_bytes_per_rhs),
        krylov_basis_bytes_per_rhs=krylov_basis_bytes_per_rhs,
        preconditioner_storage_bytes=(
            0 if preconditioner_cost is None else preconditioner_cost.storage_bytes
        ),
        preconditioner_preparation_workspace_bytes=(
            0
            if preconditioner_cost is None
            else preconditioner_cost.preparation_workspace_bytes
        ),
        preconditioner_apply_workspace_bytes_per_rhs=(
            0
            if preconditioner_cost is None
            else preconditioner_cost.apply_workspace_bytes_per_rhs
        ),
        preconditioner_setup_matvec_count=(
            0 if preconditioner_cost is None else preconditioner_cost.setup_matvec_count
        ),
        recycling_capacity=(0 if policy.recycling is None else policy.recycling.capacity),
        recycling_state_bytes=_recycling_state_bytes(problem, policy),
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
    rhs_width: int,
    /,
) -> int:
    rows = problem.operator.target.size
    columns = problem.operator.source.size
    if isinstance(problem, LeastSquaresProblem) and problem.regularizer is not None:
        rows += problem.regularizer.target.size
    if isinstance(method, (FGMRES, GMRES)):
        restart = method.restart
        basis_itemsize = (
            itemsize
            if policy.precision is None or policy.precision.krylov_dtype is None
            else jnp.dtype(policy.precision.krylov_dtype).itemsize
        )
        preconditioned_entries = (
            0 if policy.preconditioning is None else restart * columns
        )
        basis_entries = (restart + 1) * columns + preconditioned_entries
        hessenberg_entries = (restart + 1) * restart
        primal = basis_entries * basis_itemsize + hessenberg_entries * itemsize
    elif isinstance(method, BlockGMRES):
        max_steps = policy.tolerance.max_steps or columns
        restart = min(method.restart, max_steps)
        block_width = min(columns, rhs_width)
        total_entries = (
            (2 * restart + 1) * columns * block_width
            + (restart + 1) * restart * block_width * block_width
            + (2 * restart + 1) * block_width * rhs_width
        )
        primal = ((total_entries + rhs_width - 1) // rhs_width) * itemsize
    elif isinstance(method, BlockCG):
        block_width = min(columns, rhs_width)
        total_entries = (
            10 * columns * block_width
            + 6 * block_width * block_width
            + 2 * block_width * rhs_width
        )
        primal = ((total_entries + rhs_width - 1) // rhs_width) * itemsize
    elif isinstance(method, GeneralizedLSMR):
        primal = (5 * columns + 2 * rows) * itemsize
    elif isinstance(method, LSMR):
        primal = (5 * columns + 2 * rows) * itemsize
    elif isinstance(method, MINRES):
        primal = 10 * columns * itemsize
    elif isinstance(method, (PCG, ProjectedPCG, ConjugateGradient)):
        primal = 6 * columns * itemsize
    elif isinstance(method, BiCGStab):
        primal = 10 * columns * itemsize
    else:
        primal = 0
    batch_count = prod(problem.operator.batch_shape or (1,))
    recycling = _recycling_krylov_bytes(problem, method, policy, itemsize)
    if policy.differentiation.mode not in ("mathematical", "rhs-only"):
        return batch_count * max(primal, recycling)
    if isinstance(method, (BlockCG, BlockGMRES)):
        # Native block differentiation uses one full unrestarted scalar Krylov
        # basis per tangent column, independent of the primal block-step limit.
        restart = columns
    else:
        max_steps = policy.tolerance.max_steps or columns
        restart = min(30, max_steps, columns)
    tangent = ((2 * restart + 1) * columns + (restart + 1) * restart) * itemsize
    return batch_count * max(primal, tangent, recycling)


def _recycling_krylov_bytes(
    problem: AbstractLinearProblem,
    method: AbstractLinearMethod,
    policy: LinearSolvePolicy,
    itemsize: int,
    /,
) -> int:
    recycling = policy.recycling
    if recycling is None or not isinstance(method, (GMRES, FGMRES)):
        return 0
    source_size = problem.operator.source.size
    target_size = problem.operator.target.size
    max_steps = policy.tolerance.max_steps or source_size
    restart = min(method.restart, max_steps, source_size)
    capacity = recycling.capacity
    search_width = capacity + restart

    # Extraction retains an Arnoldi decomposition, its image basis, concatenated
    # retained/search bases, and the worst-case real harmonic-Ritz candidate pool.
    coordinate_entries = (
        (restart + 2) * source_size
        + (restart + 1) * restart
        + target_size * restart
        + (source_size + target_size) * search_width
        + 5 * (source_size + target_size) * capacity
    )
    # pinv/eig promote real inputs to complex values. Eight square work arrays
    # conservatively cover Gram, coupling, pseudoinverse, harmonic, and spectral
    # workspaces; the rectangular term covers selected Ritz coefficients.
    dtype = _coordinate_dtype(problem.operator.source)
    spectral_itemsize = (
        dtype.itemsize
        if jnp.issubdtype(dtype, jnp.complexfloating)
        else 2 * dtype.itemsize
    )
    spectral_entries = 8 * search_width * search_width + search_width * capacity
    return int(coordinate_entries * itemsize + spectral_entries * spectral_itemsize)


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


def _recycling_state_bytes(
    problem: AbstractLinearProblem,
    policy: LinearSolvePolicy,
    /,
) -> int:
    recycling = policy.recycling
    if recycling is None:
        return 0
    operator = problem.operator
    itemsize = _coordinate_dtype(operator.source).itemsize
    basis_bytes = (
        (operator.source.size + operator.target.size) * recycling.capacity * itemsize
    )
    scalar_bytes = 4 * jnp.dtype(jnp.int32).itemsize
    return int(basis_bytes + scalar_bytes)


def _require_selected_resources(
    estimate: LinearCostEstimate,
    policy: LinearSolvePolicy,
    rhs_count: int = 1,
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
            estimate.preparation_workspace_bytes
            + estimate.preconditioner_preparation_workspace_bytes,
            policy.resources.workspace_bytes,
        ),
        (
            "solve workspace",
            rhs_count
            * (
                estimate.solve_workspace_bytes_per_rhs
                + estimate.operator_apply_workspace_bytes_per_rhs
                + estimate.preconditioner_apply_workspace_bytes_per_rhs
            ),
            policy.resources.workspace_bytes,
        ),
        (
            "Krylov basis",
            rhs_count * estimate.krylov_basis_bytes_per_rhs,
            policy.resources.krylov_basis_bytes,
        ),
        (
            "preconditioner state",
            estimate.preconditioner_storage_bytes,
            policy.resources.preconditioner_bytes,
        ),
        (
            "recycling state",
            estimate.recycling_state_bytes,
            policy.resources.recycling_state_bytes,
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
    elif isinstance(method, BlockGMRES):
        configuration["restart"] = method.restart
    elif isinstance(method, (LSMR, GeneralizedLSMR)):
        configuration["condition_limit"] = method.condition_limit
        configuration["damping"] = method.damping
    elif isinstance(method, (SparseQR, SparseLU, SparseCholesky, SparseLDLT)):
        configuration["provider"] = method.provider
        if isinstance(method, SparseQR):
            configuration["reorder"] = method.reorder
        if isinstance(method, SparseLDLT):
            configuration["reordering"] = method.reordering
            configuration["memory_mode"] = method.memory_mode
            configuration["refinement_steps"] = method.refinement_steps
    return configuration


__all__ = ["LinearBackend", "LinearSolvePlan", "plan"]
