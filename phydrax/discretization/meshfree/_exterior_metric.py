#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Sparse, rank-aware graph metric moment problems, without metric clipping."""

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import cast, final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ...linalg import (
    ArraySpace,
    DifferentiationPolicy,
    FailurePolicy,
    GMRES,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    prepare,
    prepare_sparse_factorization,
    PreparedLinearSolve,
    PropertyEvidence,
    RankPolicy,
    refresh,
    refresh_sparse_factorization_values,
    solve,
    SparseFactorizationPlan,
    SparseFactorizationPolicy,
    SparseFactorizationStatus,
    TolerancePolicy,
)
from ...linalg._sparse_rank import (
    prepare_sparse_row_rank,
    SparseRowRankEvidence,
    SparseRowRankPolicy,
)
from ...optim import (
    ConicProgram,
    ConvexProgramResult,
    ConvexSolvePolicy,
    ConvexTermination,
    NativeHomogeneousConic,
    NonnegativeCone,
    prepare_convex_program,
    PreparedConvexProgram,
    ProductCone,
    solve_prepared_convex_program,
    ZeroCone,
)
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ...sparse._linear import _SparseStoragePlan
from ...typing import Bool, Dim, Float64, Int32, parse, Scalar, Size


class _MetricEdgeDim(Dim):
    """Compact undirected metric edges."""


class _MomentRowDim(Dim):
    """All local moment equations, including dependent equations."""


class _EquationNodeDim(Dim):
    """Nodes at which polynomial moment equations are requested."""


class _IndependentMomentDim(Dim):
    """Rows in the prepared native tolerance-defined rank profile."""


class _ReducedMomentRouteDim(Dim):
    """Moment routes belonging to the selected equations."""


class _SchurProductRouteDim(Dim):
    """Shared-edge coefficient products in sparse Schur assembly."""


MetricSign: TypeAlias = Literal["signed", "nonnegative"]
MetricAcceptance: TypeAlias = Literal["exact", "relaxed"]


class MeshfreeMetricStatus(IntEnum):
    ACCEPTED = 0
    INCOMPATIBLE_MOMENTS = 1
    PROVIDER_FAILURE = 2
    AMPLIFICATION_LIMIT = 3
    NONFINITE = 4
    RELAXED = 5


class MeshfreeMomentRowStatus(IntEnum):
    SATISFIED = 0
    INCOMPATIBLE = 1
    RELAXED = 2


@final
class MeshfreeMetricPolicy(StrictModule):
    """Exact moment constraints, or an explicitly penalized slack problem.

    Nonnegative metrics use a native sparse conic program. Its forward provider
    does not supply an automatic active-set derivative; this is reported rather
    than silently differentiating the executed optimization iterations.
    """

    __strict_contract__ = True
    sign: MetricSign = eqx.field(static=True)
    acceptance: MetricAcceptance = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    amplification_limit: float = eqx.field(static=True)
    slack_penalty: float = eqx.field(static=True)
    maximum_symbolic_entries: int = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    conic: ConvexSolvePolicy

    def __init__(
        self,
        sign: MetricSign = "signed",
        *,
        acceptance: MetricAcceptance = "exact",
        tolerance: float = 1e-9,
        rank_tolerance: float = 1e-12,
        amplification_limit: float = 1e8,
        slack_penalty: float = 1e4,
        maximum_symbolic_entries: int = 2_000_000,
        maximum_steps: int = 4096,
        conic: ConvexSolvePolicy | None = None,
    ) -> None:
        sign_ = parse(sign, MetricSign, "sign")
        acceptance_ = parse(acceptance, MetricAcceptance, "acceptance")
        numbers = (tolerance, rank_tolerance, amplification_limit, slack_penalty)
        if any(not isfinite(float(v)) or v <= 0 for v in numbers):
            raise ValueError(
                "Metric tolerances, amplification and slack penalty must be positive."
            )
        if maximum_symbolic_entries < 1 or maximum_steps < 1:
            raise ValueError(
                "Metric symbolic capacity and iteration bound must be positive."
            )
        conic_ = (
            ConvexSolvePolicy(
                NativeHomogeneousConic(
                    primal_step=min(0.05, 0.5 / slack_penalty)
                    if acceptance_ == "relaxed"
                    else 0.05,
                    dual_step=0.05,
                ),
                termination=ConvexTermination(
                    absolute=tolerance, maximum_steps=maximum_steps
                ),
                failure=FailurePolicy("status"),
            )
            if conic is None
            else conic
        )
        if not conic_.method.capabilities.sparse or conic_.regularization != 0:
            raise ValueError(
                "Metric conic provider must support sparse data without objective jitter."
            )
        self.sign = sign_
        self.acceptance = acceptance_
        self.tolerance = float(tolerance)
        self.rank_tolerance = float(rank_tolerance)
        self.amplification_limit = float(amplification_limit)
        self.slack_penalty = float(slack_penalty)
        self.maximum_symbolic_entries = int(maximum_symbolic_entries)
        self.maximum_steps = int(maximum_steps)
        self.conic = conic_


@final
class MeshfreeMetricResult(StrictModule):
    __strict_contract__ = True
    weights: Float64[_MetricEdgeDim]
    moment_residual: Float64[_MomentRowDim]
    normalized_residual: Float64[_MomentRowDim]
    slack: Float64[_MomentRowDim]
    row_status: Int32[_MomentRowDim]
    node_feasible: Bool[_EquationNodeDim]
    first_moment_residual: Float64[_EquationNodeDim]
    second_moment_residual: Float64[_EquationNodeDim]
    amplification: Float64[_EquationNodeDim]
    negative_count: Int32[Scalar]
    zero_count: Int32[Scalar]
    rank: int = eqx.field(static=True)
    redundant_constraints: int = eqx.field(static=True)
    rank_threshold: float = eqx.field(static=True)
    rank_evidence: SparseRowRankEvidence
    status: Int32[Scalar]
    provider_status: Int32[Scalar]
    accepted: Bool[Scalar]
    exact: Bool[Scalar]
    nonnegative: Bool[Scalar]
    derivative_available: Bool[Scalar]
    rank_profile_stable: Bool[Scalar]
    schur_spd: Bool[Scalar]
    schur_assessed: Bool[Scalar]
    conic_result: ConvexProgramResult | None
    derivative_contract: str = eqx.field(static=True)

    @property
    def hilbert_admitted(self) -> Array:
        return self.accepted & jnp.all(jnp.isfinite(self.weights) & (self.weights > 0))


def _coordinate_operator(
    relation: EdgeRelation,
    values: Array,
    source: ArraySpace,
    target: ArraySpace,
    storage: _SparseStoragePlan,
    *,
    symmetric: bool = False,
    positive: bool = False,
) -> SparseCoordinateOperator:
    evidence: dict[str, PropertyEvidence] = {}
    if symmetric:
        evidence["self_adjoint"] = "construction"
    if positive:
        evidence["positive_semidefinite"] = "construction"
    return SparseCoordinateOperator(
        relation,
        values,
        source=source,
        target=target,
        storage_plan=storage,
        properties=OperatorProperties(
            self_adjoint=symmetric,
            positive_semidefinite=positive,
            evidence=evidence,
        ),
    )


@final
class PreparedMeshfreeMetric(StrictModule):
    """One immutable sparse moment/rank pattern with differentiable numeric data."""

    __strict_contract__ = True
    constraint: SparseCoordinateOperator
    rhs: Float64[_MomentRowDim]
    row_scaling: Float64[_MomentRowDim]
    prior: Float64[_MetricEdgeDim]
    policy: MeshfreeMetricPolicy
    independent_rows: Int32[_IndependentMomentDim]
    selected_routes: Int32[_ReducedMomentRouteDim]
    reduced_relation: EdgeRelation
    reduced_storage: _SparseStoragePlan
    reduced_space: ArraySpace
    schur_relation: EdgeRelation | None
    schur_storage: _SparseStoragePlan | None
    schur_left: Int32[_SchurProductRouteDim] | None
    schur_right: Int32[_SchurProductRouteDim] | None
    schur_edges: Int32[_SchurProductRouteDim] | None
    schur_factor_plan: SparseFactorizationPlan | None
    linear_prepared: PreparedLinearSolve | None
    conic_relation: EdgeRelation
    conic_storage: _SparseStoragePlan
    conic_source: ArraySpace
    conic_target: ArraySpace
    objective_relation: EdgeRelation
    objective_storage: _SparseStoragePlan
    conic_prepared: PreparedConvexProgram | None
    equation_count: int = eqx.field(static=True)
    moment_count: int = eqx.field(static=True)
    intrinsic_dimension: int = eqx.field(static=True)
    rank: Size[_IndependentMomentDim] = eqx.field(static=True)
    rank_threshold: float = eqx.field(static=True)
    rank_evidence: SparseRowRankEvidence

    def __init__(
        self,
        constraint: SparseCoordinateOperator,
        rhs: ArrayLike,
        row_scaling: ArrayLike,
        prior: ArrayLike,
        policy: MeshfreeMetricPolicy,
        *,
        equation_count: int,
        intrinsic_dimension: int,
    ) -> None:
        """Prepare rank and provider structure without changing numeric values.

        This host-only assembly retains one row ordering through rank selection,
        signed Schur construction, and conic preparation; splitting those stages
        must preserve resource-refusal precedence and exact sparse identity.
        """
        rhs_ = jnp.asarray(rhs, dtype=jnp.float64)
        scaling_ = jnp.asarray(row_scaling, dtype=jnp.float64)
        prior_ = jnp.asarray(prior, dtype=jnp.float64)
        moment_count = (
            intrinsic_dimension + intrinsic_dimension * (intrinsic_dimension + 1) // 2
        )
        relation = constraint.relation
        if not isinstance(relation, EdgeRelation):
            raise TypeError(
                "Moment constraints require native edge-list sparse coordinates."
            )
        if not isinstance(constraint.source, ArraySpace) or not isinstance(
            constraint.target, ArraySpace
        ):
            raise TypeError(
                "Moment constraints require native array-coordinate source and target spaces."
            )
        if len(constraint.source.shape) != 1 or len(constraint.target.shape) != 1:
            raise ValueError("Moment spaces must use one edge and one equation vector.")
        if equation_count < 1 or intrinsic_dimension not in (1, 2, 3):
            raise ValueError(
                "Moment preparation requires equation nodes and intrinsic dimension 1, 2 or 3."
            )
        if rhs_.shape != (equation_count * moment_count,) or scaling_.shape != rhs_.shape:
            raise ValueError(
                "Moment RHS and scaling must match equation nodes and moments."
            )
        if (
            prior_.shape != (constraint.source.size,)
            or constraint.target.size != rhs_.size
        ):
            raise ValueError("Moment design, RHS and edge prior dimensions disagree.")
        if not np.all(np.isfinite(np.asarray(scaling_)) & (np.asarray(scaling_) > 0)):
            raise ValueError("Moment row scaling must be finite and positive.")
        if not np.all(np.isfinite(np.asarray(prior_)) & (np.asarray(prior_) > 0)):
            raise ValueError("Moment edge prior must be finite and positive.")
        rows = np.asarray(relation.target_indices)
        columns = np.asarray(relation.source_indices)
        coefficients = np.asarray(
            constraint.coefficients / scaling_[relation.target_indices]
        )
        if not np.all(np.isfinite(coefficients)) or not np.all(
            np.isfinite(np.asarray(rhs_))
        ):
            raise ValueError("Prepared moment data must be finite.")
        normalized_design = eqx.tree_at(
            lambda op: op.coefficients,
            constraint,
            jnp.asarray(coefficients, dtype=jnp.float64),
        )
        rank_evidence = prepare_sparse_row_rank(
            normalized_design,
            SparseRowRankPolicy(
                RankPolicy(relative_cutoff=policy.rank_tolerance),
                maximum_rows=policy.maximum_symbolic_entries,
                maximum_input_nonzeros=policy.maximum_symbolic_entries,
                maximum_factor_nonzeros=policy.maximum_symbolic_entries,
                maximum_elimination_work=policy.maximum_symbolic_entries,
            ),
        )
        independent = np.asarray(rank_evidence.selected_rows)
        threshold = float(np.asarray(rank_evidence.pivot_threshold))
        rank = rank_evidence.rank
        if rank == 0:
            raise ValueError(
                "Moment design has zero rank; no diffusion metric is definable."
            )
        relaxed = policy.acceptance == "relaxed"
        used_rows = (
            np.arange(constraint.target.size, dtype=np.int32) if relaxed else independent
        )
        inverse = np.full(constraint.target.size, -1, dtype=np.int32)
        inverse[used_rows] = np.arange(used_rows.size, dtype=np.int32)
        routes = np.flatnonzero(inverse[rows] >= 0).astype(np.int32)
        reduced_space = ArraySpace(
            (used_rows.size,),
            dtype=jnp.float64,
            space_id=f"{constraint.target.space_id}:independent",
        )
        reduced_relation = EdgeRelation(
            columns[routes],
            inverse[rows[routes]],
            source_size=constraint.source.size,
            target_size=used_rows.size,
        )
        reduced_storage = _SparseStoragePlan(reduced_relation)
        schur_relation: EdgeRelation | None = None
        schur_storage: _SparseStoragePlan | None = None
        schur_left: Array | None = None
        schur_right: Array | None = None
        schur_edges: Array | None = None
        factor_plan: SparseFactorizationPlan | None = None
        linear_prepared: PreparedLinearSolve | None = None
        if policy.sign == "signed":
            by_edge: list[list[int]] = [[] for _ in range(constraint.source.size)]
            for route, edge in enumerate(columns[routes]):
                by_edge[int(edge)].append(route)
            left: list[int] = []
            right: list[int] = []
            edge_indices: list[int] = []
            for edge, indices in enumerate(by_edge):
                if len(left) + len(indices) ** 2 > policy.maximum_symbolic_entries:
                    raise ValueError(
                        "Sparse meshfree Schur assembly exceeds symbolic capacity."
                    )
                for i in indices:
                    for j in indices:
                        left.append(i)
                        right.append(j)
                        edge_indices.append(edge)
            schur_left = jnp.asarray(left, dtype=jnp.int32)
            schur_right = jnp.asarray(right, dtype=jnp.int32)
            schur_edges = jnp.asarray(edge_indices, dtype=jnp.int32)
            schur_rows = inverse[rows[routes]][np.asarray(left)]
            schur_columns = inverse[rows[routes]][np.asarray(right)]
            if relaxed:
                schur_rows = np.concatenate((schur_rows, np.arange(used_rows.size)))
                schur_columns = np.concatenate((schur_columns, np.arange(used_rows.size)))
            schur_relation = EdgeRelation(
                schur_columns,
                schur_rows,
                source_size=used_rows.size,
                target_size=used_rows.size,
            )
            schur_storage = _SparseStoragePlan(schur_relation)
            seed = _coordinate_operator(
                schur_relation,
                jnp.ones(schur_rows.shape, dtype=jnp.float64),
                reduced_space,
                reduced_space,
                schur_storage,
                symmetric=True,
            )
            factor_plan = prepare_sparse_factorization(
                seed,
                SparseFactorizationPolicy(
                    "cholesky", max_symbolic_work=policy.maximum_symbolic_entries
                ),
            )
            linear_prepared = prepare(
                LinearSystem(seed),
                LinearSolvePolicy(
                    GMRES(restart=min(64, used_rows.size)),
                    tolerance=TolerancePolicy(
                        relative=policy.tolerance,
                        absolute=policy.tolerance,
                        max_steps=policy.maximum_steps,
                    ),
                    differentiation=DifferentiationPolicy("mathematical"),
                    failure=FailurePolicy("status"),
                ),
            )
        edges = constraint.source.size
        moments = used_rows.size
        variables = edges + (moments if relaxed else 0)
        conic_rows = inverse[rows[routes]]
        conic_columns = columns[routes]
        if relaxed:
            conic_rows = np.concatenate((conic_rows, np.arange(moments)))
            conic_columns = np.concatenate((conic_columns, edges + np.arange(moments)))
        conic_rows = np.concatenate((conic_rows, moments + np.arange(edges)))
        conic_columns = np.concatenate((conic_columns, np.arange(edges)))
        conic_relation = EdgeRelation(
            conic_columns,
            conic_rows,
            source_size=variables,
            target_size=moments + edges,
        )
        conic_storage = _SparseStoragePlan(conic_relation)
        conic_source = ArraySpace(
            (variables,),
            dtype=jnp.float64,
            space_id=f"{constraint.source.space_id}:conic",
        )
        conic_target = ArraySpace(
            (moments + edges,),
            dtype=jnp.float64,
            space_id=f"{constraint.target.space_id}:conic",
        )
        objective_relation = EdgeRelation(
            np.arange(variables),
            np.arange(variables),
            source_size=variables,
            target_size=variables,
        )
        objective_storage = _SparseStoragePlan(objective_relation)
        conic_prepared: PreparedConvexProgram | None = None
        if policy.sign == "nonnegative":
            initial_values = jnp.asarray(coefficients[routes], dtype=jnp.float64)
            initial_quadratic = 1 / prior_
            if relaxed:
                initial_values = jnp.concatenate((initial_values, jnp.ones((moments,))))
                initial_quadratic = jnp.concatenate(
                    (initial_quadratic, jnp.full((moments,), policy.slack_penalty))
                )
            initial_values = jnp.concatenate((initial_values, -jnp.ones((edges,))))
            matrix = _coordinate_operator(
                conic_relation, initial_values, conic_source, conic_target, conic_storage
            )
            quadratic = _coordinate_operator(
                objective_relation,
                initial_quadratic,
                conic_source,
                conic_source,
                objective_storage,
                symmetric=True,
                positive=True,
            )
            program = ConicProgram(
                quadratic,
                jnp.zeros((variables,), dtype=jnp.float64),
                matrix,
                jnp.concatenate(
                    ((rhs_ / scaling_)[used_rows], jnp.zeros((edges,), dtype=jnp.float64))
                ),
                ProductCone((ZeroCone(moments), NonnegativeCone(edges))),
                problem_id=f"{constraint.operator_id}:nonnegative",
                convexity_evidence="construction",
            )
            conic_prepared = prepare_convex_program(program, policy.conic)
        self.constraint = constraint
        self.rhs = rhs_
        self.row_scaling = scaling_
        self.prior = prior_
        self.policy = policy
        self.equation_count = int(equation_count)
        self.intrinsic_dimension = int(intrinsic_dimension)
        self.moment_count = moment_count
        self.rank = rank
        self.rank_threshold = threshold
        self.rank_evidence = rank_evidence
        self.independent_rows = jnp.asarray(independent)
        self.selected_routes = jnp.asarray(routes)
        self.reduced_space = reduced_space
        self.reduced_relation = reduced_relation
        self.reduced_storage = reduced_storage
        self.schur_left = schur_left
        self.schur_right = schur_right
        self.schur_edges = schur_edges
        self.schur_relation = schur_relation
        self.schur_storage = schur_storage
        self.schur_factor_plan = factor_plan
        self.linear_prepared = linear_prepared
        self.conic_relation = conic_relation
        self.conic_storage = conic_storage
        self.conic_source = conic_source
        self.conic_target = conic_target
        self.objective_relation = objective_relation
        self.objective_storage = objective_storage
        self.conic_prepared = conic_prepared

    def solve(
        self,
        *,
        prior: ArrayLike | None = None,
        rhs: ArrayLike | None = None,
        coefficients: ArrayLike | None = None,
    ) -> MeshfreeMetricResult:
        """Solve and retain the complete metric admission evidence.

        Signed and conic branches share the final physical moment assessment.
        Keep their reduction order and failure evidence explicit rather than
        hiding provider-dependent admission in small forwarding helpers.
        """
        phi = self.prior if prior is None else jnp.asarray(prior, dtype=jnp.float64)
        raw_rhs = self.rhs if rhs is None else jnp.asarray(rhs, dtype=jnp.float64)
        raw_values = (
            self.constraint.coefficients
            if coefficients is None
            else jnp.asarray(coefficients, dtype=jnp.float64)
        )
        if (
            phi.shape != self.prior.shape
            or raw_rhs.shape != self.rhs.shape
            or raw_values.shape != self.constraint.coefficients.shape
        ):
            raise ValueError(
                "Metric refresh must preserve the prepared sparse dimensions."
            )
        phi = eqx.error_if(
            phi,
            jnp.any(~jnp.isfinite(phi) | (phi <= 0)),
            "Metric prior must be finite and strictly positive.",
        )
        relation = cast(EdgeRelation, self.constraint.relation)
        values = raw_values / self.row_scaling[relation.target_indices]
        reduced_values = values[self.selected_routes]
        normalized_rhs = raw_rhs / self.row_scaling
        relaxed = self.policy.acceptance == "relaxed"
        reduced_rhs = normalized_rhs if relaxed else normalized_rhs[self.independent_rows]
        schur_spd = jnp.asarray(False)
        schur_assessed = jnp.asarray(self.policy.sign == "signed")
        conic_result: ConvexProgramResult | None = None
        if self.policy.sign == "signed":
            left, right, edges = self.schur_left, self.schur_right, self.schur_edges
            schur_relation, schur_storage = self.schur_relation, self.schur_storage
            factor_plan, linear_prepared = self.schur_factor_plan, self.linear_prepared
            if (
                left is None
                or right is None
                or edges is None
                or schur_relation is None
                or schur_storage is None
                or factor_plan is None
                or linear_prepared is None
            ):
                raise RuntimeError(
                    "Signed metric requires its prepared native Schur solve."
                )
            reduced = _coordinate_operator(
                self.reduced_relation,
                reduced_values,
                cast(ArraySpace, self.constraint.source),
                self.reduced_space,
                self.reduced_storage,
            )
            schur_values = reduced_values[left] * phi[edges] * reduced_values[right]
            if relaxed:
                schur_values = jnp.concatenate(
                    (
                        schur_values,
                        jnp.full(reduced_rhs.shape, 1 / self.policy.slack_penalty),
                    )
                )
            schur = _coordinate_operator(
                schur_relation,
                schur_values,
                self.reduced_space,
                self.reduced_space,
                schur_storage,
                symmetric=True,
            )
            factor = refresh_sparse_factorization_values(
                factor_plan, schur.sparse_storage().values
            )
            schur_spd = factor.status == int(SparseFactorizationStatus.SUCCESS)
            linear_result = solve(
                refresh(linear_prepared, LinearSystem(schur)), reduced_rhs
            )
            weights = phi * reduced.transpose_mv(linear_result.value)
            provider_status = linear_result.status.astype(jnp.int32)
            provider_success = linear_result.successful & schur_spd
            derivative = provider_success
        else:
            conic_values = reduced_values
            quadratic_values = 1 / phi
            if relaxed:
                conic_values = jnp.concatenate(
                    (conic_values, jnp.ones(reduced_rhs.shape))
                )
                quadratic_values = jnp.concatenate(
                    (
                        quadratic_values,
                        jnp.full(reduced_rhs.shape, self.policy.slack_penalty),
                    )
                )
            conic_values = jnp.concatenate((conic_values, -jnp.ones(phi.shape)))
            quadratic = _coordinate_operator(
                self.objective_relation,
                quadratic_values,
                self.conic_source,
                self.conic_source,
                self.objective_storage,
                symmetric=True,
                positive=True,
            )
            matrix = _coordinate_operator(
                self.conic_relation,
                conic_values,
                self.conic_source,
                self.conic_target,
                self.conic_storage,
            )
            if self.conic_prepared is None:
                raise RuntimeError(
                    "Nonnegative metric has no prepared native conic lifecycle."
                )
            program = eqx.tree_at(
                lambda p: (p.quadratic, p.constraint_matrix, p.constraint_rhs),
                self.conic_prepared.program,
                (quadratic, matrix, jnp.concatenate((reduced_rhs, jnp.zeros(phi.shape)))),
            )
            bound = PreparedConvexProgram(
                program,
                self.conic_prepared.template,
                numeric_version=self.conic_prepared.numeric_version + 1,
                numeric_binding_id=f"{self.conic_prepared.numeric_binding_id}:metric-refresh",
            )
            conic_result = solve_prepared_convex_program(bound).result
            weights = conic_result.primal[: phi.size]
            provider_status = conic_result.status.astype(jnp.int32)
            provider_success = conic_result.successful
            derivative = jnp.asarray(False)
        raw_constraint = eqx.tree_at(
            lambda op: op.coefficients, self.constraint, raw_values
        )
        residual = raw_constraint.mv(weights) - raw_rhs
        normalized = residual / self.row_scaling
        thresholds = self.policy.tolerance * (1 + jnp.abs(normalized_rhs))
        row_ok = jnp.abs(normalized) <= thresholds
        node_ok = jnp.all(
            row_ok.reshape((self.equation_count, self.moment_count)), axis=1
        )
        moment_matrix = residual.reshape((self.equation_count, self.moment_count))
        first = jnp.max(jnp.abs(moment_matrix[:, : self.intrinsic_dimension]), axis=1)
        second = jnp.max(jnp.abs(moment_matrix[:, self.intrinsic_dimension :]), axis=1)
        absolute_moment = eqx.tree_at(
            lambda op: op.coefficients, raw_constraint, jnp.abs(raw_values)
        ).mv(jnp.abs(weights))
        amplification = jnp.max(
            (absolute_moment / (self.row_scaling + jnp.abs(raw_rhs))).reshape(
                (self.equation_count, self.moment_count)
            ),
            axis=1,
        )
        finite = jnp.all(jnp.isfinite(weights)) & jnp.all(jnp.isfinite(residual))
        amplification_ok = jnp.all(amplification <= self.policy.amplification_limit)
        exact = jnp.all(row_ok)
        nonnegative = jnp.all(weights >= 0)
        sign_ok = nonnegative if self.policy.sign == "nonnegative" else jnp.asarray(True)
        accepted = (
            finite
            & provider_success
            & amplification_ok
            & sign_ok
            & (jnp.asarray(relaxed) | exact)
        )
        status = jnp.where(
            ~finite,
            int(MeshfreeMetricStatus.NONFINITE),
            jnp.where(
                ~provider_success | ~sign_ok,
                int(MeshfreeMetricStatus.PROVIDER_FAILURE),
                jnp.where(
                    ~amplification_ok,
                    int(MeshfreeMetricStatus.AMPLIFICATION_LIMIT),
                    jnp.where(
                        ~exact & ~jnp.asarray(relaxed),
                        int(MeshfreeMetricStatus.INCOMPATIBLE_MOMENTS),
                        int(
                            MeshfreeMetricStatus.RELAXED
                            if relaxed
                            else MeshfreeMetricStatus.ACCEPTED
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        # Selected-row SPD proves a rank lower bound, not absence of newly
        # independent discarded equations. Maximal rank closes that gap.
        # A relaxed all-equation slack system instead has a constructive SPD
        # Schur for every finite C. Other profiles expose no generic coordinate
        # derivative admission, even when the current moments are compatible.
        rank_profile_stable = schur_spd & jnp.asarray(
            relaxed
            or self.rank == min(self.constraint.target.size, self.constraint.source.size),
        )
        return MeshfreeMetricResult(
            weights=weights,
            moment_residual=residual,
            normalized_residual=normalized,
            slack=residual if relaxed else jnp.zeros_like(residual),
            row_status=jnp.where(
                row_ok,
                int(MeshfreeMomentRowStatus.SATISFIED),
                int(
                    MeshfreeMomentRowStatus.RELAXED
                    if relaxed
                    else MeshfreeMomentRowStatus.INCOMPATIBLE
                ),
            ).astype(jnp.int32),
            node_feasible=node_ok,
            first_moment_residual=first,
            second_moment_residual=second,
            amplification=amplification,
            negative_count=jnp.sum(weights < 0, dtype=jnp.int32),
            zero_count=jnp.sum(weights == 0, dtype=jnp.int32),
            rank=self.rank,
            redundant_constraints=self.constraint.target.size - self.rank,
            rank_threshold=self.rank_threshold,
            rank_evidence=self.rank_evidence,
            status=status,
            provider_status=provider_status,
            accepted=accepted,
            exact=exact,
            nonnegative=nonnegative,
            derivative_available=derivative & accepted & rank_profile_stable,
            rank_profile_stable=rank_profile_stable,
            schur_spd=schur_spd,
            schur_assessed=schur_assessed,
            conic_result=conic_result,
            derivative_contract=(
                "all-equation penalized-slack implicit Schur"
                if relaxed
                else "fixed maximal-rank compatible-tangent implicit Schur"
            )
            if self.policy.sign == "signed"
            else "active-set derivative unavailable from selected forward provider",
        )
