#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Resource-bounded quotient-algebra roots for small affine polynomial systems."""

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.linalg as scipy_linalg
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    DifferentiationPolicy,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    solve as solve_linear,
)
from ..linalg.eigen import (
    DenseSchurQZ,
    general_eigensolve,
    GeneralEigenproblem,
    GeneralEigenResourcePolicy,
    GeneralEigenSolvePolicy,
    GeneralEigenTolerancePolicy,
)
from ..nonlinear import SmallRootKernel
from ._system import SparsePolynomialSupport, SparsePolynomialSystem


class QuotientRootStatus(IntEnum):
    """Fail-closed outcome for a bounded quotient-algebra root calculation."""

    SUCCESS = 0
    RESOURCE_REJECTED = 1
    SUPPORT_NOT_AFFINE = 2
    RANK_AMBIGUOUS = 3
    QUOTIENT_CLOSURE_AMBIGUOUS = 4
    QUOTIENT_BASIS_ILL_CONDITIONED = 5
    MULTIPLICATION_SOLVE_FAILED = 6
    COMMUTATOR_AMBIGUOUS = 7
    EIGEN_RECOVERY_AMBIGUOUS = 8
    REPEATED_ROOT_AMBIGUITY = 9
    NONFINITE_RESULT = 10
    ORIGINAL_RESIDUAL_REJECTED = 11


def quotient_root_status_name(status: int | Array, /) -> str:
    """Return the stable lower-case name of one quotient-root status."""

    return QuotientRootStatus(int(np.asarray(status))).name.lower()


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _nonnegative_finite(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


class QuotientRootResourcePolicy(StrictModule):
    """Hard limits checked before any Macaulay monomial allocation."""

    maximum_variables: int = eqx.field(static=True)
    maximum_equations: int = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    maximum_monomials: int = eqx.field(static=True)
    maximum_macaulay_rows: int = eqx.field(static=True)
    maximum_assembly_entries: int = eqx.field(static=True)
    maximum_dense_entries: int = eqx.field(static=True)
    maximum_quotient_dimension: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_variables: int = 8,
        maximum_equations: int = 8,
        maximum_degree: int = 16,
        maximum_monomials: int = 4096,
        maximum_macaulay_rows: int = 4096,
        maximum_assembly_entries: int = 1_000_000,
        maximum_dense_entries: int = 4_000_000,
        maximum_quotient_dimension: int = 256,
    ):
        values = tuple(
            _positive_integer(value, name)
            for value, name in (
                (maximum_variables, "maximum_variables"),
                (maximum_equations, "maximum_equations"),
                (maximum_degree, "maximum_degree"),
                (maximum_monomials, "maximum_monomials"),
                (maximum_macaulay_rows, "maximum_macaulay_rows"),
                (maximum_assembly_entries, "maximum_assembly_entries"),
                (maximum_dense_entries, "maximum_dense_entries"),
                (maximum_quotient_dimension, "maximum_quotient_dimension"),
            )
        )
        (
            self.maximum_variables,
            self.maximum_equations,
            self.maximum_degree,
            self.maximum_monomials,
            self.maximum_macaulay_rows,
            self.maximum_assembly_entries,
            self.maximum_dense_entries,
            self.maximum_quotient_dimension,
        ) = values
        self.policy_id = canonical_fingerprint(
            {"kind": "quotient-root-resources", "limits": list(values)}
        )


class QuotientRankPolicy(StrictModule):
    """Two-threshold rank decision and conditioning acceptance policy."""

    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    ambiguity_factor: float = eqx.field(static=True)
    maximum_basis_condition: float = eqx.field(static=True)
    maximum_eigenvalue_condition: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative_tolerance: float = 1e-9,
        absolute_tolerance: float = 1e-12,
        ambiguity_factor: float = 8.0,
        maximum_basis_condition: float = 1e10,
        maximum_eigenvalue_condition: float = 1e10,
    ):
        relative = _nonnegative_finite(relative_tolerance, "relative_tolerance")
        absolute = _nonnegative_finite(absolute_tolerance, "absolute_tolerance")
        ambiguity = float(ambiguity_factor)
        basis_condition = float(maximum_basis_condition)
        eigen_condition = float(maximum_eigenvalue_condition)
        if not math.isfinite(ambiguity) or ambiguity <= 1.0:
            raise ValueError("ambiguity_factor must be finite and greater than one.")
        if not math.isfinite(basis_condition) or basis_condition <= 1.0:
            raise ValueError(
                "maximum_basis_condition must be finite and greater than one."
            )
        if not math.isfinite(eigen_condition) or eigen_condition <= 1.0:
            raise ValueError(
                "maximum_eigenvalue_condition must be finite and greater than one."
            )
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.ambiguity_factor = ambiguity
        self.maximum_basis_condition = basis_condition
        self.maximum_eigenvalue_condition = eigen_condition
        self.policy_id = canonical_fingerprint(
            {
                "kind": "quotient-rank-policy",
                "relative_tolerance": float(relative).hex(),
                "absolute_tolerance": float(absolute).hex(),
                "ambiguity_factor": ambiguity.hex(),
                "maximum_basis_condition": basis_condition.hex(),
                "maximum_eigenvalue_condition": eigen_condition.hex(),
            }
        )


class QuotientRootPolicy(StrictModule):
    """Numerical acceptance policy separate from the finite resource envelope."""

    resources: QuotientRootResourcePolicy
    rank: QuotientRankPolicy
    commutator_tolerance: float = eqx.field(static=True)
    triangularization_tolerance: float = eqx.field(static=True)
    cluster_absolute_tolerance: float = eqx.field(static=True)
    cluster_relative_tolerance: float = eqx.field(static=True)
    residual_absolute_tolerance: float = eqx.field(static=True)
    residual_relative_tolerance: float = eqx.field(static=True)
    simple_root_relative_tolerance: float = eqx.field(static=True)
    polish_maximum_steps: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        resources: QuotientRootResourcePolicy | None = None,
        rank: QuotientRankPolicy | None = None,
        commutator_tolerance: float = 1e-7,
        triangularization_tolerance: float = 1e-7,
        cluster_absolute_tolerance: float = 1e-8,
        cluster_relative_tolerance: float = 1e-7,
        residual_absolute_tolerance: float = 1e-7,
        residual_relative_tolerance: float = 1e-7,
        simple_root_relative_tolerance: float = 1e-7,
        polish_maximum_steps: int = 12,
    ):
        resources_ = QuotientRootResourcePolicy() if resources is None else resources
        rank_ = QuotientRankPolicy() if rank is None else rank
        if not isinstance(resources_, QuotientRootResourcePolicy):
            raise TypeError("resources must be QuotientRootResourcePolicy or None.")
        if not isinstance(rank_, QuotientRankPolicy):
            raise TypeError("rank must be QuotientRankPolicy or None.")
        tolerances = tuple(
            _nonnegative_finite(value, name)
            for value, name in (
                (commutator_tolerance, "commutator_tolerance"),
                (triangularization_tolerance, "triangularization_tolerance"),
                (cluster_absolute_tolerance, "cluster_absolute_tolerance"),
                (cluster_relative_tolerance, "cluster_relative_tolerance"),
                (residual_absolute_tolerance, "residual_absolute_tolerance"),
                (residual_relative_tolerance, "residual_relative_tolerance"),
                (simple_root_relative_tolerance, "simple_root_relative_tolerance"),
            )
        )
        steps = _positive_integer(polish_maximum_steps, "polish_maximum_steps")
        self.resources = resources_
        self.rank = rank_
        (
            self.commutator_tolerance,
            self.triangularization_tolerance,
            self.cluster_absolute_tolerance,
            self.cluster_relative_tolerance,
            self.residual_absolute_tolerance,
            self.residual_relative_tolerance,
            self.simple_root_relative_tolerance,
        ) = tolerances
        self.polish_maximum_steps = steps
        self.policy_id = canonical_fingerprint(
            {
                "kind": "quotient-root-policy",
                "resources": resources_.policy_id,
                "rank": rank_.policy_id,
                "tolerances": [float(value).hex() for value in tolerances],
                "polish_maximum_steps": steps,
            }
        )


class QuotientRootPlan(StrictModule):
    """Support-only Macaulay layout with allocation estimates and admission status."""

    policy: QuotientRootPolicy
    monomials: Array
    equation_degrees: Array
    row_equations: Array
    row_multipliers: Array
    assembly_rows: Array
    assembly_columns: Array
    assembly_terms: Array
    variable_count: int = eqx.field(static=True)
    equation_count: int = eqx.field(static=True)
    macaulay_degree: int = eqx.field(static=True)
    monomial_count: int = eqx.field(static=True)
    macaulay_row_count: int = eqx.field(static=True)
    assembly_entry_count: int = eqx.field(static=True)
    dense_entry_count: int = eqx.field(static=True)
    estimated_dense_bytes: int = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    status: int = eqx.field(static=True)
    rejection_reasons: tuple[str, ...] = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class QuotientPreparationEvidence(StrictModule):
    """Numerical path evidence; it does not assert that recovered roots are physical."""

    singular_values: Array
    definite_rank: Array
    possible_rank: Array
    basis_condition: Array
    multiplication_solve_residual: Array
    commutator_error: Array
    path_accepted: Array
    evidence_id: str = eqx.field(static=True)


class PreparedQuotientRootSolver(StrictModule):
    """Coefficient-bound quotient algebra, refreshable on one fixed support."""

    system: SparsePolynomialSystem
    plan: QuotientRootPlan
    macaulay_matrix: Array
    quotient_basis: Array
    multiplication_matrices: Array
    preparation: QuotientPreparationEvidence
    numeric_version: Array
    refresh_count: Array
    quotient_dimension: int = eqx.field(static=True)
    status: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class QuotientRootEvidence(StrictModule):
    """Recovery-path diagnostics kept separate from original-system replay."""

    preparation: QuotientPreparationEvidence
    joint_weights: Array
    joint_spectrum_separation: Array
    triangularization_error: Array
    maximum_eigenvalue_condition: Array
    original_residual_tolerance: Array
    original_residual_maximum: Array
    residual_accepted: Array
    root_count: int = eqx.field(static=True)
    cluster_count: int = eqx.field(static=True)
    eigen_backend: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class QuotientRootResult(StrictModule):
    """Recovered candidates, ambiguity state, and original-system replay evidence."""

    roots: Array
    original_residuals: Array
    residual_norms: Array
    multiplication_matrices: Array
    cluster_labels: Array
    cluster_multiplicities: Array
    polished: Array
    polish_status: Array
    status: Array
    path_accepted: Array
    residual_accepted: Array
    evidence: QuotientRootEvidence
    support_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return (
            (self.status == int(QuotientRootStatus.SUCCESS))
            & self.path_accepted
            & self.residual_accepted
        )

    @property
    def root_count(self) -> int:
        return int(self.roots.shape[0])

    def replay(self, system: SparsePolynomialSystem, /) -> Array:
        """Replay candidates through a coefficient-compatible original system."""

        if not isinstance(system, SparsePolynomialSystem):
            raise TypeError("system must be SparsePolynomialSystem.")
        if system.support.support_id != self.support_id:
            raise ValueError("Replay system support differs from the result support.")
        return system.evaluate(self.roots)


def _exact_degree_monomials(
    variable_count: int, degree: int, prefix: tuple[int, ...] = ()
):
    if variable_count == 1:
        yield prefix + (degree,)
        return
    for leading in range(degree, -1, -1):
        yield from _exact_degree_monomials(
            variable_count - 1,
            degree - leading,
            prefix + (leading,),
        )


def enumerate_monomials(
    variable_count: int, maximum_degree: int, /
) -> tuple[tuple[int, ...], ...]:
    """Enumerate multivariate monomials in deterministic graded lexicographic order."""

    variables = _positive_integer(variable_count, "variable_count")
    if isinstance(maximum_degree, bool) or not isinstance(maximum_degree, int):
        raise TypeError("maximum_degree must be an integer.")
    if maximum_degree < 0:
        raise ValueError("maximum_degree must be nonnegative.")
    return tuple(
        monomial
        for degree in range(maximum_degree + 1)
        for monomial in _exact_degree_monomials(variables, degree)
    )


def _monomial_count(variable_count: int, degree: int, /) -> int:
    return math.comb(variable_count + degree, degree) if degree >= 0 else 0


def _equation_degrees(support: SparsePolynomialSupport, /) -> tuple[int, ...]:
    equations = np.asarray(support.equation_indices, dtype=np.int64)
    exponents = np.asarray(support.exponents, dtype=np.int64)
    return tuple(
        int(np.max(np.sum(exponents[equations == equation], axis=1), initial=0))
        for equation in range(support.equation_count)
    )


def _support_is_affine(support: SparsePolynomialSupport, /) -> bool:
    return all(group.geometry == "affine" for group in support.groups)


def plan_quotient_roots(
    support: SparsePolynomialSupport,
    policy: QuotientRootPolicy | None = None,
    /,
    *,
    macaulay_degree: int | None = None,
) -> QuotientRootPlan:
    """Plan bounded Macaulay assembly without inspecting polynomial coefficients."""

    if not isinstance(support, SparsePolynomialSupport):
        raise TypeError("support must be SparsePolynomialSupport.")
    policy_ = QuotientRootPolicy() if policy is None else policy
    if not isinstance(policy_, QuotientRootPolicy):
        raise TypeError("policy must be QuotientRootPolicy or None.")
    degrees = _equation_degrees(support)
    inferred_degree = max(
        max(degrees, default=0),
        1 + sum(max(degree - 1, 0) for degree in degrees),
    )
    if macaulay_degree is None:
        degree = inferred_degree
    else:
        if isinstance(macaulay_degree, bool) or not isinstance(macaulay_degree, int):
            raise TypeError("macaulay_degree must be an integer or None.")
        if macaulay_degree < 0:
            raise ValueError("macaulay_degree must be nonnegative.")
        degree = macaulay_degree
    variable_count = support.variable_count
    equation_count = support.equation_count
    monomial_count = _monomial_count(variable_count, degree)
    multiplier_counts = tuple(
        _monomial_count(variable_count, degree - equation_degree)
        for equation_degree in degrees
    )
    row_count = sum(multiplier_counts)
    equation_indices = np.asarray(support.equation_indices, dtype=np.int64)
    term_counts = tuple(
        int(np.count_nonzero(equation_indices == equation))
        for equation in range(equation_count)
    )
    assembly_count = sum(
        multiplier_count * term_count
        for multiplier_count, term_count in zip(
            multiplier_counts, term_counts, strict=True
        )
    )
    dense_count = monomial_count * row_count
    resources = policy_.resources
    reasons: list[str] = []
    if not _support_is_affine(support):
        reasons.append("projective-variable-group")
    for exceeded, reason in (
        (variable_count > resources.maximum_variables, "variable-count"),
        (equation_count > resources.maximum_equations, "equation-count"),
        (degree > resources.maximum_degree, "macaulay-degree"),
        (monomial_count > resources.maximum_monomials, "monomial-count"),
        (row_count > resources.maximum_macaulay_rows, "macaulay-row-count"),
        (assembly_count > resources.maximum_assembly_entries, "assembly-entry-count"),
        (dense_count > resources.maximum_dense_entries, "dense-entry-count"),
    ):
        if exceeded:
            reasons.append(reason)
    accepted = not reasons
    if not _support_is_affine(support):
        status = QuotientRootStatus.SUPPORT_NOT_AFFINE
    elif not accepted:
        status = QuotientRootStatus.RESOURCE_REJECTED
    else:
        status = QuotientRootStatus.SUCCESS

    if accepted:
        monomial_tuple = enumerate_monomials(variable_count, degree)
        monomial_lookup = {
            monomial: index for index, monomial in enumerate(monomial_tuple)
        }
        row_equations: list[int] = []
        row_multipliers: list[tuple[int, ...]] = []
        assembly_rows: list[int] = []
        assembly_columns: list[int] = []
        assembly_terms: list[int] = []
        exponents = np.asarray(support.exponents, dtype=np.int64)
        for equation, equation_degree in enumerate(degrees):
            multipliers = enumerate_monomials(variable_count, degree - equation_degree)
            term_indices = np.flatnonzero(equation_indices == equation)
            for multiplier in multipliers:
                row = len(row_equations)
                row_equations.append(equation)
                row_multipliers.append(multiplier)
                for term in term_indices:
                    product = tuple(
                        int(exponents[term, axis]) + multiplier[axis]
                        for axis in range(variable_count)
                    )
                    assembly_rows.append(row)
                    assembly_columns.append(monomial_lookup[product])
                    assembly_terms.append(int(term))
        monomials = jnp.asarray(monomial_tuple, dtype=jnp.int32)
        row_equation_array = jnp.asarray(row_equations, dtype=jnp.int32)
        multiplier_array = jnp.asarray(row_multipliers, dtype=jnp.int32).reshape(
            (row_count, variable_count)
        )
        assembly_row_array = jnp.asarray(assembly_rows, dtype=jnp.int32)
        assembly_column_array = jnp.asarray(assembly_columns, dtype=jnp.int32)
        assembly_term_array = jnp.asarray(assembly_terms, dtype=jnp.int32)
    else:
        monomials = jnp.zeros((0, variable_count), dtype=jnp.int32)
        row_equation_array = jnp.zeros((0,), dtype=jnp.int32)
        multiplier_array = jnp.zeros((0, variable_count), dtype=jnp.int32)
        assembly_row_array = jnp.zeros((0,), dtype=jnp.int32)
        assembly_column_array = jnp.zeros((0,), dtype=jnp.int32)
        assembly_term_array = jnp.zeros((0,), dtype=jnp.int32)
    plan_id = canonical_fingerprint(
        {
            "kind": "quotient-root-plan",
            "support": support.support_id,
            "policy": policy_.policy_id,
            "macaulay_degree": degree,
            "monomial_count": monomial_count,
            "macaulay_row_count": row_count,
            "assembly_entry_count": assembly_count,
            "dense_entry_count": dense_count,
            "accepted": accepted,
            "status": int(status),
            "rejection_reasons": reasons,
        }
    )
    return QuotientRootPlan(
        policy=policy_,
        monomials=monomials,
        equation_degrees=jnp.asarray(degrees, dtype=jnp.int32),
        row_equations=row_equation_array,
        row_multipliers=multiplier_array,
        assembly_rows=assembly_row_array,
        assembly_columns=assembly_column_array,
        assembly_terms=assembly_term_array,
        variable_count=variable_count,
        equation_count=equation_count,
        macaulay_degree=degree,
        monomial_count=monomial_count,
        macaulay_row_count=row_count,
        assembly_entry_count=assembly_count,
        dense_entry_count=dense_count,
        estimated_dense_bytes=16 * dense_count,
        accepted=accepted,
        status=int(status),
        rejection_reasons=tuple(reasons),
        support_id=support.support_id,
        plan_id=plan_id,
    )


def assemble_macaulay_matrix(
    plan: QuotientRootPlan, system: SparsePolynomialSystem, /
) -> Array:
    """Assemble the planned dense Macaulay matrix from canonical sparse coefficients."""

    if not isinstance(plan, QuotientRootPlan):
        raise TypeError("plan must be QuotientRootPlan.")
    if not isinstance(system, SparsePolynomialSystem):
        raise TypeError("system must be SparsePolynomialSystem.")
    if system.support.support_id != plan.support_id:
        raise ValueError("System support does not match the quotient-root plan.")
    if not plan.accepted:
        return jnp.zeros((0, 0), dtype=system.coefficients.dtype)
    values = system.coefficients[plan.assembly_terms]
    return (
        jnp.zeros(
            (plan.macaulay_row_count, plan.monomial_count),
            dtype=system.coefficients.dtype,
        )
        .at[plan.assembly_rows, plan.assembly_columns]
        .add(values)
    )


def _rank_interval(matrix: np.ndarray, policy: QuotientRankPolicy, /):
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=True)
    if singular_values.size == 0:
        return singular_values, 0, 0, right
    scale = float(singular_values[0])
    epsilon = np.finfo(singular_values.dtype).eps
    lower = max(
        policy.absolute_tolerance,
        policy.relative_tolerance * scale,
        max(matrix.shape, default=1) * epsilon * scale,
    )
    upper = policy.ambiguity_factor * lower
    definite = int(np.count_nonzero(singular_values > upper))
    possible = int(np.count_nonzero(singular_values > lower))
    return singular_values, definite, possible, right


def _independent_basis_indices(
    nullspace: np.ndarray,
    monomials: np.ndarray,
    maximum_basis_degree: int,
    policy: QuotientRankPolicy,
    /,
) -> tuple[tuple[int, ...], float]:
    dimension = nullspace.shape[0]
    selected: list[int] = []
    for index, monomial in enumerate(monomials):
        if int(np.sum(monomial)) > maximum_basis_degree:
            continue
        candidate = nullspace[:, selected + [index]]
        singular_values = np.linalg.svd(candidate, compute_uv=False)
        scale = float(singular_values[0])
        epsilon = np.finfo(singular_values.dtype).eps
        threshold = max(
            policy.absolute_tolerance,
            policy.relative_tolerance * scale,
            max(candidate.shape) * epsilon * scale,
        )
        if float(singular_values[-1]) > threshold:
            selected.append(index)
        if len(selected) == dimension:
            break
    if len(selected) != dimension:
        return tuple(selected), math.inf
    singular_values = np.linalg.svd(nullspace[:, selected], compute_uv=False)
    condition = float(singular_values[0] / singular_values[-1])
    return tuple(selected), condition


def _preparation_evidence(
    *,
    singular_values: Array,
    definite_rank: int,
    possible_rank: int,
    basis_condition: float,
    solve_residual: float,
    commutator_error: float,
    accepted: bool,
    plan: QuotientRootPlan,
    system: SparsePolynomialSystem,
) -> QuotientPreparationEvidence:
    return QuotientPreparationEvidence(
        singular_values=singular_values,
        definite_rank=jnp.asarray(definite_rank, dtype=jnp.int32),
        possible_rank=jnp.asarray(possible_rank, dtype=jnp.int32),
        basis_condition=jnp.asarray(basis_condition),
        multiplication_solve_residual=jnp.asarray(solve_residual),
        commutator_error=jnp.asarray(commutator_error),
        path_accepted=jnp.asarray(accepted),
        evidence_id=canonical_fingerprint(
            {
                "kind": "quotient-preparation-evidence",
                "plan": plan.plan_id,
                "system": system.system_id,
                "method": "macaulay-svd-rank-reveal-native-lu",
            }
        ),
    )


def _failed_prepared(
    system: SparsePolynomialSystem,
    plan: QuotientRootPlan,
    status: QuotientRootStatus,
    macaulay: Array,
    *,
    singular_values: Array | None = None,
    definite_rank: int = 0,
    possible_rank: int = 0,
    basis_condition: float = math.inf,
    solve_residual: float = math.inf,
    commutator_error: float = math.inf,
    numeric_version: int = 0,
    refresh_count: int = 0,
    prepared_id: str | None = None,
) -> PreparedQuotientRootSolver:
    identifier = (
        canonical_fingerprint({"kind": "prepared-quotient-root", "plan": plan.plan_id})
        if prepared_id is None
        else prepared_id
    )
    return PreparedQuotientRootSolver(
        system=system,
        plan=plan,
        macaulay_matrix=macaulay,
        quotient_basis=jnp.zeros((0, plan.variable_count), dtype=jnp.int32),
        multiplication_matrices=jnp.zeros(
            (plan.variable_count, 0, 0), dtype=system.coefficients.dtype
        ),
        preparation=_preparation_evidence(
            singular_values=(
                jnp.zeros((0,), dtype=jnp.asarray(macaulay).real.dtype)
                if singular_values is None
                else singular_values
            ),
            definite_rank=definite_rank,
            possible_rank=possible_rank,
            basis_condition=basis_condition,
            solve_residual=solve_residual,
            commutator_error=commutator_error,
            accepted=False,
            plan=plan,
            system=system,
        ),
        numeric_version=jnp.asarray(numeric_version, dtype=jnp.int32),
        refresh_count=jnp.asarray(refresh_count, dtype=jnp.int32),
        quotient_dimension=0,
        status=int(status),
        prepared_id=identifier,
    )


def _commutator_error(matrices: np.ndarray, /) -> float:
    maximum = 0.0
    for left in range(matrices.shape[0]):
        for right in range(left + 1, matrices.shape[0]):
            product_scale = max(
                1.0,
                float(np.linalg.norm(matrices[left]))
                * float(np.linalg.norm(matrices[right])),
            )
            commutator = (
                matrices[left] @ matrices[right] - matrices[right] @ matrices[left]
            )
            maximum = max(maximum, float(np.linalg.norm(commutator)) / product_scale)
    return maximum


def _prepare_numeric(
    system: SparsePolynomialSystem,
    plan: QuotientRootPlan,
    /,
    *,
    numeric_version: int,
    refresh_count: int,
    prepared_id: str | None,
) -> PreparedQuotientRootSolver:
    if not plan.accepted:
        return _failed_prepared(
            system,
            plan,
            QuotientRootStatus(plan.status),
            jnp.zeros((0, 0), dtype=system.coefficients.dtype),
            numeric_version=numeric_version,
            refresh_count=refresh_count,
            prepared_id=prepared_id,
        )
    macaulay = assemble_macaulay_matrix(plan, system)
    host_matrix = np.asarray(macaulay)
    if not np.all(np.isfinite(host_matrix)):
        return _failed_prepared(
            system,
            plan,
            QuotientRootStatus.NONFINITE_RESULT,
            macaulay,
            numeric_version=numeric_version,
            refresh_count=refresh_count,
            prepared_id=prepared_id,
        )
    singular_values, definite_rank, possible_rank, right = _rank_interval(
        host_matrix, plan.policy.rank
    )
    singular_array = jnp.asarray(singular_values)
    if definite_rank != possible_rank:
        return _failed_prepared(
            system,
            plan,
            QuotientRootStatus.RANK_AMBIGUOUS,
            macaulay,
            singular_values=singular_array,
            definite_rank=definite_rank,
            possible_rank=possible_rank,
            numeric_version=numeric_version,
            refresh_count=refresh_count,
            prepared_id=prepared_id,
        )
    rank = definite_rank
    quotient_dimension = plan.monomial_count - rank
    if (
        quotient_dimension < 1
        or quotient_dimension > plan.policy.resources.maximum_quotient_dimension
    ):
        return _failed_prepared(
            system,
            plan,
            QuotientRootStatus.QUOTIENT_CLOSURE_AMBIGUOUS,
            macaulay,
            singular_values=singular_array,
            definite_rank=rank,
            possible_rank=rank,
            numeric_version=numeric_version,
            refresh_count=refresh_count,
            prepared_id=prepared_id,
        )
    monomials = np.asarray(plan.monomials, dtype=np.int64)
    nullspace = right[rank:, :]
    basis_indices, basis_condition = _independent_basis_indices(
        nullspace,
        monomials,
        plan.macaulay_degree - 1,
        plan.policy.rank,
    )
    if len(basis_indices) != quotient_dimension:
        return _failed_prepared(
            system,
            plan,
            QuotientRootStatus.QUOTIENT_CLOSURE_AMBIGUOUS,
            macaulay,
            singular_values=singular_array,
            definite_rank=rank,
            possible_rank=rank,
            basis_condition=basis_condition,
            numeric_version=numeric_version,
            refresh_count=refresh_count,
            prepared_id=prepared_id,
        )
    if (
        not math.isfinite(basis_condition)
        or basis_condition > plan.policy.rank.maximum_basis_condition
    ):
        return _failed_prepared(
            system,
            plan,
            QuotientRootStatus.QUOTIENT_BASIS_ILL_CONDITIONED,
            macaulay,
            singular_values=singular_array,
            definite_rank=rank,
            possible_rank=rank,
            basis_condition=basis_condition,
            numeric_version=numeric_version,
            refresh_count=refresh_count,
            prepared_id=prepared_id,
        )
    basis_set = set(basis_indices)
    pivot_indices = tuple(
        index for index in range(plan.monomial_count) if index not in basis_set
    )
    row_basis = right[:rank, :]
    pivot_matrix = row_basis[:, pivot_indices]
    basis_matrix = row_basis[:, basis_indices]
    if rank:
        linear_result = solve_linear(
            LinearSystem(DenseLinearOperator(jnp.asarray(pivot_matrix))),
            -jnp.asarray(basis_matrix),
            policy=LinearSolvePolicy(
                DenseLU(),
                differentiation=DifferentiationPolicy("none"),
                failure=FailurePolicy("status"),
            ),
        )
        pivot_reductions = np.asarray(linear_result.value)
        solve_successful = bool(np.all(np.asarray(linear_result.successful)))
        solve_residual = float(
            np.linalg.norm(pivot_matrix @ pivot_reductions + basis_matrix)
            / max(1.0, float(np.linalg.norm(basis_matrix)))
        )
    else:
        pivot_reductions = np.zeros((0, quotient_dimension), dtype=host_matrix.dtype)
        solve_successful = True
        solve_residual = 0.0
    if not solve_successful or not np.all(np.isfinite(pivot_reductions)):
        return _failed_prepared(
            system,
            plan,
            QuotientRootStatus.MULTIPLICATION_SOLVE_FAILED,
            macaulay,
            singular_values=singular_array,
            definite_rank=rank,
            possible_rank=rank,
            basis_condition=basis_condition,
            solve_residual=solve_residual,
            numeric_version=numeric_version,
            refresh_count=refresh_count,
            prepared_id=prepared_id,
        )
    basis_lookup = {index: column for column, index in enumerate(basis_indices)}
    pivot_lookup = {index: row for row, index in enumerate(pivot_indices)}
    monomial_lookup = {tuple(value): index for index, value in enumerate(monomials)}
    multiplication = np.zeros(
        (plan.variable_count, quotient_dimension, quotient_dimension),
        dtype=np.result_type(host_matrix, complex),
    )
    closure_valid = True
    for variable in range(plan.variable_count):
        increment = np.zeros((plan.variable_count,), dtype=np.int64)
        increment[variable] = 1
        for basis_column, monomial_index in enumerate(basis_indices):
            product = tuple(monomials[monomial_index] + increment)
            product_index = monomial_lookup.get(product)
            if product_index is None:
                closure_valid = False
                continue
            basis_row = basis_lookup.get(product_index)
            if basis_row is not None:
                multiplication[variable, basis_row, basis_column] = 1.0
            else:
                pivot_row = pivot_lookup.get(product_index)
                if pivot_row is None:
                    closure_valid = False
                else:
                    multiplication[variable, :, basis_column] = pivot_reductions[
                        pivot_row
                    ]
    if not closure_valid:
        return _failed_prepared(
            system,
            plan,
            QuotientRootStatus.QUOTIENT_CLOSURE_AMBIGUOUS,
            macaulay,
            singular_values=singular_array,
            definite_rank=rank,
            possible_rank=rank,
            basis_condition=basis_condition,
            solve_residual=solve_residual,
            numeric_version=numeric_version,
            refresh_count=refresh_count,
            prepared_id=prepared_id,
        )
    commutator_error = _commutator_error(multiplication)
    status = (
        QuotientRootStatus.COMMUTATOR_AMBIGUOUS
        if commutator_error > plan.policy.commutator_tolerance
        else QuotientRootStatus.SUCCESS
    )
    identifier = (
        canonical_fingerprint({"kind": "prepared-quotient-root", "plan": plan.plan_id})
        if prepared_id is None
        else prepared_id
    )
    return PreparedQuotientRootSolver(
        system=system,
        plan=plan,
        macaulay_matrix=macaulay,
        quotient_basis=jnp.asarray(monomials[list(basis_indices)], dtype=jnp.int32),
        multiplication_matrices=jnp.asarray(multiplication),
        preparation=_preparation_evidence(
            singular_values=singular_array,
            definite_rank=rank,
            possible_rank=rank,
            basis_condition=basis_condition,
            solve_residual=solve_residual,
            commutator_error=commutator_error,
            accepted=status is QuotientRootStatus.SUCCESS,
            plan=plan,
            system=system,
        ),
        numeric_version=jnp.asarray(numeric_version, dtype=jnp.int32),
        refresh_count=jnp.asarray(refresh_count, dtype=jnp.int32),
        quotient_dimension=quotient_dimension,
        status=int(status),
        prepared_id=identifier,
    )


def prepare_quotient_root_solver(
    system: SparsePolynomialSystem,
    plan: QuotientRootPlan | QuotientRootPolicy | None = None,
    /,
) -> PreparedQuotientRootSolver:
    """Bind coefficients to a support plan and construct multiplication operators."""

    if not isinstance(system, SparsePolynomialSystem):
        raise TypeError("system must be SparsePolynomialSystem.")
    if isinstance(plan, QuotientRootPlan):
        selected = plan
    elif plan is None or isinstance(plan, QuotientRootPolicy):
        selected = plan_quotient_roots(system.support, plan)
    else:
        raise TypeError("plan must be QuotientRootPlan, QuotientRootPolicy, or None.")
    if selected.support_id != system.support.support_id:
        raise ValueError("System support does not match the quotient-root plan.")
    return _prepare_numeric(
        system,
        selected,
        numeric_version=0,
        refresh_count=0,
        prepared_id=None,
    )


def refresh_quotient_root_solver(
    prepared: PreparedQuotientRootSolver,
    system: SparsePolynomialSystem,
    /,
) -> PreparedQuotientRootSolver:
    """Refresh coefficients while preserving the exact support plan and path identity."""

    if not isinstance(prepared, PreparedQuotientRootSolver):
        raise TypeError("prepared must be PreparedQuotientRootSolver.")
    if not isinstance(system, SparsePolynomialSystem):
        raise TypeError("system must be SparsePolynomialSystem.")
    if system.support.support_id != prepared.plan.support_id:
        raise ValueError("Coefficient refresh requires exactly the planned support.")
    return _prepare_numeric(
        system,
        prepared.plan,
        numeric_version=int(np.asarray(prepared.numeric_version)) + 1,
        refresh_count=int(np.asarray(prepared.refresh_count)) + 1,
        prepared_id=prepared.prepared_id,
    )


def _joint_weight_candidates(variable_count: int, /) -> tuple[np.ndarray, ...]:
    indices = np.arange(1, variable_count + 1, dtype=float)
    return tuple(
        np.exp(1j * phase * indices) / math.sqrt(variable_count)
        for phase in (math.sqrt(2.0), math.sqrt(3.0), math.sqrt(5.0))
    )


def _minimum_separation(values: np.ndarray, /) -> float:
    if values.size < 2:
        return math.inf
    difference = np.abs(values[:, None] - values[None, :])
    difference[np.diag_indices(values.size)] = np.inf
    return float(np.min(difference))


def _joint_schur_recovery(matrices: np.ndarray, /):
    best = None
    best_scaled_separation = -math.inf
    for weights in _joint_weight_candidates(matrices.shape[0]):
        joint = np.einsum("v,vij->ij", weights, matrices)
        triangular, schur_vectors = scipy_linalg.schur(
            joint,
            output="complex",
            check_finite=False,
        )
        joint_values = np.diag(triangular)
        scale = max(1.0, float(np.max(np.abs(joint_values), initial=0.0)))
        separation = _minimum_separation(joint_values)
        scaled = separation / scale
        if scaled > best_scaled_separation:
            transformed = np.asarray(
                [schur_vectors.conj().T @ matrix @ schur_vectors for matrix in matrices]
            )
            roots = np.stack([np.diag(matrix) for matrix in transformed], axis=-1)
            lower_error = max(
                (
                    float(np.linalg.norm(np.tril(matrix, k=-1)))
                    / max(1.0, float(np.linalg.norm(matrix)))
                    for matrix in transformed
                ),
                default=0.0,
            )
            best = (roots, weights, separation, lower_error, joint)
            best_scaled_separation = scaled
    assert best is not None
    return best


def _lexicographic_root_order(roots: np.ndarray, /) -> np.ndarray:
    keys: list[np.ndarray] = []
    for variable in range(roots.shape[1] - 1, -1, -1):
        keys.extend((roots[:, variable].imag, roots[:, variable].real))
    return np.lexsort(tuple(keys))


def _cluster_roots(
    roots: np.ndarray,
    absolute_tolerance: float,
    relative_tolerance: float,
    /,
) -> tuple[np.ndarray, np.ndarray, int]:
    count = roots.shape[0]
    parent = np.arange(count, dtype=np.int32)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for left in range(count):
        for right in range(left + 1, count):
            scale = max(
                1.0,
                float(np.max(np.abs(roots[left]), initial=0.0)),
                float(np.max(np.abs(roots[right]), initial=0.0)),
            )
            distance = float(np.max(np.abs(roots[left] - roots[right]), initial=0.0))
            if distance <= absolute_tolerance + relative_tolerance * scale:
                left_root = root(left)
                right_root = root(right)
                if left_root != right_root:
                    parent[right_root] = left_root
    labels = np.zeros((count,), dtype=np.int32)
    canonical: dict[int, int] = {}
    for index in range(count):
        representative = root(index)
        if representative not in canonical:
            canonical[representative] = len(canonical)
        labels[index] = canonical[representative]
    multiplicities = np.asarray(
        [np.count_nonzero(labels == label) for label in labels], dtype=np.int32
    )
    return labels, multiplicities, len(canonical)


def _maximum_eigenvalue_condition(
    joint: np.ndarray, dimension: int, /
) -> tuple[float, bool]:
    problem = GeneralEigenproblem(DenseLinearOperator(jnp.asarray(joint)))
    solved = general_eigensolve(
        problem,
        policy=GeneralEigenSolvePolicy(
            DenseSchurQZ(),
            tolerance=GeneralEigenTolerancePolicy(
                relative=1e-7,
                absolute=1e-9,
                biorthogonality=1e-6,
                cluster_relative=1e-7,
            ),
            resources=GeneralEigenResourcePolicy(max_dimension=max(1, dimension)),
            failure=FailurePolicy("status"),
        ),
    )
    conditions = np.asarray(solved.diagnostics.eigenvalue_condition_estimates)
    maximum = float(np.max(conditions, initial=0.0))
    return maximum, bool(np.asarray(solved.successful))


def _simple_regular_mask(
    system: SparsePolynomialSystem,
    roots: np.ndarray,
    cluster_multiplicities: np.ndarray,
    tolerance: float,
    /,
) -> np.ndarray:
    if system.support.equation_count != system.support.variable_count:
        return np.zeros((roots.shape[0],), dtype=bool)
    jacobians = np.asarray(system.jacobian(jnp.asarray(roots)))
    result = np.zeros((roots.shape[0],), dtype=bool)
    for index, jacobian in enumerate(jacobians):
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        scale = float(singular_values[0])
        result[index] = (
            cluster_multiplicities[index] == 1
            and np.all(np.isfinite(singular_values))
            and float(singular_values[-1]) > tolerance * max(1.0, scale)
        )
    return result


def _polish_simple_roots(
    system: SparsePolynomialSystem,
    roots: np.ndarray,
    eligible: np.ndarray,
    policy: QuotientRootPolicy,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(eligible)
    polished = np.zeros((roots.shape[0],), dtype=bool)
    statuses = np.full((roots.shape[0],), -1, dtype=np.int32)
    if indices.size == 0:
        return roots, polished, statuses
    variable_count = system.support.variable_count

    def embedded_residual(state, _):
        point = state[:variable_count] + 1j * state[variable_count:]
        residual = system.evaluate(point)
        return jnp.concatenate((jnp.real(residual), jnp.imag(residual)))

    initial = np.concatenate((roots[indices].real, roots[indices].imag), axis=1)
    solved = SmallRootKernel(
        embedded_residual,
        maximum_dimension=2 * variable_count,
        maximum_steps=policy.polish_maximum_steps,
        absolute_tolerance=policy.residual_absolute_tolerance,
        relative_tolerance=policy.residual_relative_tolerance,
    ).solve(jnp.asarray(initial), None)
    solved_states = np.asarray(solved.state)
    candidates = (
        solved_states[:, :variable_count] + 1j * solved_states[:, variable_count:]
    )
    successful = np.asarray(solved.successful, dtype=bool)
    output = roots.copy()
    output[indices[successful]] = candidates[successful]
    polished[indices[successful]] = True
    statuses[indices] = np.asarray(solved.status, dtype=np.int32)
    return output, polished, statuses


def _empty_result(
    prepared: PreparedQuotientRootSolver,
    /,
    *,
    status: QuotientRootStatus | None = None,
) -> QuotientRootResult:
    system = prepared.system
    outcome = prepared.status if status is None else int(status)
    roots = jnp.zeros(
        (0, prepared.plan.variable_count),
        dtype=jnp.result_type(system.coefficients.dtype, jnp.complex64),
    )
    residuals = jnp.zeros((0, prepared.plan.equation_count), dtype=roots.dtype)
    status_array = jnp.asarray(outcome, dtype=jnp.int32)
    evidence_id = canonical_fingerprint(
        {
            "kind": "quotient-root-evidence",
            "prepared": prepared.prepared_id,
            "system": system.system_id,
            "numeric_version": int(np.asarray(prepared.numeric_version)),
            "status": outcome,
        }
    )
    evidence = QuotientRootEvidence(
        preparation=prepared.preparation,
        joint_weights=jnp.zeros((prepared.plan.variable_count,), dtype=roots.dtype),
        joint_spectrum_separation=jnp.asarray(0.0),
        triangularization_error=jnp.asarray(math.inf),
        maximum_eigenvalue_condition=jnp.asarray(math.inf),
        original_residual_tolerance=jnp.asarray(0.0),
        original_residual_maximum=jnp.asarray(math.inf),
        residual_accepted=jnp.asarray(False),
        root_count=0,
        cluster_count=0,
        eigen_backend="not-run",
        evidence_id=evidence_id,
    )
    return QuotientRootResult(
        roots=roots,
        original_residuals=residuals,
        residual_norms=jnp.zeros((0,), dtype=roots.real.dtype),
        multiplication_matrices=prepared.multiplication_matrices,
        cluster_labels=jnp.zeros((0,), dtype=jnp.int32),
        cluster_multiplicities=jnp.zeros((0,), dtype=jnp.int32),
        polished=jnp.zeros((0,), dtype=bool),
        polish_status=jnp.zeros((0,), dtype=jnp.int32),
        status=status_array,
        path_accepted=jnp.asarray(False),
        residual_accepted=jnp.asarray(False),
        evidence=evidence,
        support_id=prepared.plan.support_id,
        system_id=system.system_id,
        plan_id=prepared.plan.plan_id,
        result_id=canonical_fingerprint(
            {"kind": "quotient-root-result", "evidence": evidence_id}
        ),
    )


def solve_quotient_roots(
    system_or_prepared: SparsePolynomialSystem | PreparedQuotientRootSolver,
    /,
    *,
    plan: QuotientRootPlan | QuotientRootPolicy | None = None,
    polish: bool = False,
) -> QuotientRootResult:
    """Recover finite root candidates and replay them through the original system.

    A successful status is deliberately narrower than finding plausible candidates:
    rank, quotient closure, commutation, joint spectral separation, conditioning, and
    original residual replay must all pass the declared policy.
    """

    if isinstance(system_or_prepared, PreparedQuotientRootSolver):
        if plan is not None:
            raise ValueError("plan must be omitted for a prepared quotient solver.")
        prepared = system_or_prepared
    elif isinstance(system_or_prepared, SparsePolynomialSystem):
        prepared = prepare_quotient_root_solver(system_or_prepared, plan)
    else:
        raise TypeError("Expected SparsePolynomialSystem or PreparedQuotientRootSolver.")
    if not isinstance(polish, bool):
        raise TypeError("polish must be a bool.")
    recoverable = prepared.status in (
        int(QuotientRootStatus.SUCCESS),
        int(QuotientRootStatus.COMMUTATOR_AMBIGUOUS),
    )
    if not recoverable or prepared.quotient_dimension == 0:
        return _empty_result(prepared)

    policy = prepared.plan.policy
    matrices = np.asarray(prepared.multiplication_matrices)
    try:
        roots, weights, separation, triangular_error, joint = _joint_schur_recovery(
            matrices
        )
    except np.linalg.LinAlgError:
        return _empty_result(
            prepared,
            status=QuotientRootStatus.EIGEN_RECOVERY_AMBIGUOUS,
        )
    order = _lexicographic_root_order(roots)
    roots = roots[order]
    labels, multiplicities, cluster_count = _cluster_roots(
        roots,
        policy.cluster_absolute_tolerance,
        policy.cluster_relative_tolerance,
    )
    repeated = bool(np.any(multiplicities > 1))
    if repeated:
        maximum_condition, eigen_successful = math.inf, False
    else:
        maximum_condition, eigen_successful = _maximum_eigenvalue_condition(
            joint, prepared.quotient_dimension
        )
    finite = bool(
        np.all(np.isfinite(roots))
        and np.isfinite(triangular_error)
        and (prepared.quotient_dimension == 1 or np.isfinite(separation))
    )
    if repeated:
        status = QuotientRootStatus.REPEATED_ROOT_AMBIGUITY
    elif not finite:
        status = QuotientRootStatus.NONFINITE_RESULT
    elif (
        not eigen_successful
        or triangular_error > policy.triangularization_tolerance
        or maximum_condition > policy.rank.maximum_eigenvalue_condition
    ):
        status = QuotientRootStatus.EIGEN_RECOVERY_AMBIGUOUS
    elif prepared.status == int(QuotientRootStatus.COMMUTATOR_AMBIGUOUS):
        status = QuotientRootStatus.COMMUTATOR_AMBIGUOUS
    else:
        status = QuotientRootStatus.SUCCESS

    polished = np.zeros((roots.shape[0],), dtype=bool)
    polish_status = np.full((roots.shape[0],), -1, dtype=np.int32)
    if polish and status is QuotientRootStatus.SUCCESS:
        eligible = _simple_regular_mask(
            prepared.system,
            roots,
            multiplicities,
            policy.simple_root_relative_tolerance,
        )
        roots, polished, polish_status = _polish_simple_roots(
            prepared.system, roots, eligible, policy
        )
        labels, multiplicities, cluster_count = _cluster_roots(
            roots,
            policy.cluster_absolute_tolerance,
            policy.cluster_relative_tolerance,
        )

    path_accepted = status is QuotientRootStatus.SUCCESS
    residuals = np.asarray(prepared.system.evaluate(jnp.asarray(roots)))
    residual_norms = np.max(np.abs(residuals), axis=-1, initial=0.0)
    coefficient_scale = max(
        1.0,
        float(
            np.max(
                np.abs(np.asarray(prepared.system.coefficients)),
                initial=0.0,
            )
        ),
    )
    residual_tolerance = (
        policy.residual_absolute_tolerance
        + policy.residual_relative_tolerance * coefficient_scale
    )
    residual_accepted = bool(
        np.all(np.isfinite(residuals)) and np.all(residual_norms <= residual_tolerance)
    )
    if status is QuotientRootStatus.SUCCESS and not residual_accepted:
        status = QuotientRootStatus.ORIGINAL_RESIDUAL_REJECTED
    evidence_id = canonical_fingerprint(
        {
            "kind": "quotient-root-evidence",
            "prepared": prepared.prepared_id,
            "system": prepared.system.system_id,
            "numeric_version": int(np.asarray(prepared.numeric_version)),
            "method": "deterministic-joint-schur",
            "status": int(status),
        }
    )
    evidence = QuotientRootEvidence(
        preparation=prepared.preparation,
        joint_weights=jnp.asarray(weights),
        joint_spectrum_separation=jnp.asarray(separation),
        triangularization_error=jnp.asarray(triangular_error),
        maximum_eigenvalue_condition=jnp.asarray(maximum_condition),
        original_residual_tolerance=jnp.asarray(residual_tolerance),
        original_residual_maximum=jnp.asarray(float(np.max(residual_norms, initial=0.0))),
        residual_accepted=jnp.asarray(residual_accepted),
        root_count=roots.shape[0],
        cluster_count=cluster_count,
        eigen_backend="phydrax-dense-schur-qz; deterministic-joint-schur",
        evidence_id=evidence_id,
    )
    return QuotientRootResult(
        roots=jnp.asarray(roots),
        original_residuals=jnp.asarray(residuals),
        residual_norms=jnp.asarray(residual_norms),
        multiplication_matrices=prepared.multiplication_matrices,
        cluster_labels=jnp.asarray(labels, dtype=jnp.int32),
        cluster_multiplicities=jnp.asarray(multiplicities, dtype=jnp.int32),
        polished=jnp.asarray(polished),
        polish_status=jnp.asarray(polish_status, dtype=jnp.int32),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        path_accepted=jnp.asarray(path_accepted),
        residual_accepted=jnp.asarray(residual_accepted),
        evidence=evidence,
        support_id=prepared.plan.support_id,
        system_id=prepared.system.system_id,
        plan_id=prepared.plan.plan_id,
        result_id=canonical_fingerprint(
            {"kind": "quotient-root-result", "evidence": evidence_id}
        ),
    )


__all__ = [
    "PreparedQuotientRootSolver",
    "QuotientPreparationEvidence",
    "QuotientRankPolicy",
    "QuotientRootEvidence",
    "QuotientRootPlan",
    "QuotientRootPolicy",
    "QuotientRootResourcePolicy",
    "QuotientRootResult",
    "QuotientRootStatus",
    "assemble_macaulay_matrix",
    "enumerate_monomials",
    "plan_quotient_roots",
    "prepare_quotient_root_solver",
    "quotient_root_status_name",
    "refresh_quotient_root_solver",
    "solve_quotient_roots",
]
