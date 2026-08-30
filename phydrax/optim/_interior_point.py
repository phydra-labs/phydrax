#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._nonlinear_precision import NonlinearPrecisionPolicy
from ..linalg import (
    DenseLU,
    LinearSolvePolicy,
    sparse_provider_capabilities,
    SparseLDLT,
)
from ._filter_ipm import (
    FilterInteriorPoint as _DenseFilterInteriorPoint,
    FilterInteriorPointEvidence,
)
from ._iterative import (
    AbstractMinimizationMethod,
    Bounds,
    ConstrainedOptimalityCertificate,
    MinimizationProblem,
    MinimizationResult,
    NonlinearConstraint,
    OptimizationCapabilities,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)
from ._primal_dual import (
    PrimalDualNewtonKrylov as _MatrixFreeCenteredInteriorPoint,
    PrimalDualPredictorCorrector as _MatrixFreePredictorCorrector,
)
from ._structured_ipm import solve_sparse_structured_ipm
from ._structured_method import (
    AbstractStructuredNonlinearMethod,
    StructuredNonlinearCapabilities,
)
from ._structured_nonlinear import (
    PreparedStructuredNonlinearProgram,
    StructuredNonlinearResult,
    StructuredNonlinearWarmStart,
    StructuredOptimizationWork,
)


InteriorPointMode: TypeAlias = Literal[
    "dense-filter",
    "matrix-free-centered",
    "matrix-free-predictor-corrector",
    "sparse-augmented",
]


def _maximum(value: Array, /) -> Array:
    return jnp.max(jnp.abs(value), initial=jnp.asarray(0.0, dtype=value.dtype))


def _canonical_structured_multipliers(
    prepared: PreparedStructuredNonlinearProgram,
    certificate: ConstrainedOptimalityCertificate | None,
    /,
) -> tuple[Array, Array, Array]:
    program = prepared.program
    dtype = prepared.variable_lower.dtype
    constraint_multipliers = jnp.zeros((program.num_constraints,), dtype=dtype)
    lower_bound_multipliers = jnp.zeros((program.num_variables,), dtype=dtype)
    upper_bound_multipliers = jnp.zeros((program.num_variables,), dtype=dtype)
    if certificate is None:
        return (
            constraint_multipliers,
            lower_bound_multipliers,
            upper_bound_multipliers,
        )

    equality_indices = np.asarray(program.equality_indices, dtype=np.int32)
    lower_indices = np.asarray(program.lower_indices, dtype=np.int32)
    upper_indices = np.asarray(program.upper_indices, dtype=np.int32)
    lower_x = np.isfinite(np.asarray(prepared.variable_lower))
    upper_x = np.isfinite(np.asarray(prepared.variable_upper))
    fixed_x = (
        lower_x
        & upper_x
        & (np.asarray(prepared.variable_lower) == np.asarray(prepared.variable_upper))
    )
    nonfixed_lower_x = np.flatnonzero(lower_x & ~fixed_x)
    nonfixed_upper_x = np.flatnonzero(upper_x & ~fixed_x)
    fixed_x_indices = np.flatnonzero(fixed_x)

    equality_count = int(equality_indices.size)
    lower_constraint_count = int(lower_indices.size)
    upper_constraint_count = int(upper_indices.size)
    equality_values = certificate.equality_multipliers
    inequality_values = certificate.inequality_multipliers
    expected_equalities = equality_count + int(fixed_x_indices.size)
    expected_inequalities = (
        lower_constraint_count
        + int(nonfixed_lower_x.size)
        + upper_constraint_count
        + int(nonfixed_upper_x.size)
    )
    if equality_values.shape != (expected_equalities,):
        raise ValueError("Interior-point equality multiplier layout changed.")
    if inequality_values.shape != (expected_inequalities,):
        raise ValueError("Interior-point inequality multiplier layout changed.")

    constraint_multipliers = constraint_multipliers.at[jnp.asarray(equality_indices)].set(
        equality_values[:equality_count]
    )
    fixed_values = equality_values[equality_count:]
    lower_fixed = jnp.maximum(-fixed_values, 0.0)
    upper_fixed = jnp.maximum(fixed_values, 0.0)
    lower_bound_multipliers = lower_bound_multipliers.at[
        jnp.asarray(fixed_x_indices)
    ].set(lower_fixed)
    upper_bound_multipliers = upper_bound_multipliers.at[
        jnp.asarray(fixed_x_indices)
    ].set(upper_fixed)

    cursor = 0
    lower_constraint_values = inequality_values[cursor : cursor + lower_constraint_count]
    cursor += lower_constraint_count
    lower_variable_values = inequality_values[
        cursor : cursor + int(nonfixed_lower_x.size)
    ]
    cursor += int(nonfixed_lower_x.size)
    upper_constraint_values = inequality_values[cursor : cursor + upper_constraint_count]
    cursor += upper_constraint_count
    upper_variable_values = inequality_values[cursor:]

    constraint_multipliers = constraint_multipliers.at[jnp.asarray(lower_indices)].add(
        -lower_constraint_values
    )
    constraint_multipliers = constraint_multipliers.at[jnp.asarray(upper_indices)].add(
        upper_constraint_values
    )
    lower_bound_multipliers = lower_bound_multipliers.at[
        jnp.asarray(nonfixed_lower_x)
    ].set(lower_variable_values)
    upper_bound_multipliers = upper_bound_multipliers.at[
        jnp.asarray(nonfixed_upper_x)
    ].set(upper_variable_values)
    return (
        constraint_multipliers,
        lower_bound_multipliers,
        upper_bound_multipliers,
    )


def _updated_diagnostics(
    diagnostics: OptimizationDiagnostics,
    certificate: ConstrainedOptimalityCertificate,
    optimality: Array,
    /,
) -> OptimizationDiagnostics:
    return eqx.tree_at(
        lambda value: (
            value.final_optimality_norm,
            value.primal_feasibility,
            value.dual_feasibility,
            value.complementarity,
            value.active_constraints,
        ),
        diagnostics,
        (
            optimality,
            certificate.primal_feasibility,
            certificate.dual_feasibility,
            certificate.complementarity,
            jnp.sum(certificate.active_mask, dtype=jnp.int32),
        ),
    )


def _structured_work(
    result: MinimizationResult,
    mode: InteriorPointMode,
    /,
) -> StructuredOptimizationWork:
    diagnostics = result.diagnostics
    factorizations = jnp.asarray(0, dtype=jnp.int32)
    right_hand_sides = diagnostics.linear_solves
    restoration_evaluations = jnp.asarray(0, dtype=jnp.int32)
    if mode == "dense-filter" and isinstance(
        result.method_evidence,
        FilterInteriorPointEvidence,
    ):
        factorizations = result.method_evidence.kkt_factorizations
        right_hand_sides = result.method_evidence.kkt_rhs_solves
        restoration_evaluations = result.method_evidence.restoration_steps
    return StructuredOptimizationWork(
        objective_evaluations=diagnostics.objective_evaluations,
        constraint_evaluations=diagnostics.constraint_evaluations,
        gradient_evaluations=diagnostics.gradient_evaluations,
        jacobian_evaluations=diagnostics.jacobian_evaluations,
        hessian_evaluations=diagnostics.hvp_evaluations,
        kkt_assemblies=factorizations,
        factorizations=factorizations,
        right_hand_side_solves=right_hand_sides,
        backtracking_evaluations=diagnostics.globalization_evaluations,
        restoration_evaluations=restoration_evaluations,
        certificate_evaluations=1,
        complete=diagnostics.counts_complete,
    )


class PrimalDualInteriorPoint(AbstractStructuredNonlinearMethod):
    """One configured primal-dual interior-point method over generic or structured NLPs."""

    mode: InteriorPointMode = eqx.field(static=True)
    implementation: AbstractMinimizationMethod
    structured_linear_policy: LinearSolvePolicy
    fraction_to_boundary: float = eqx.field(static=True)
    sufficient_decrease: float = eqx.field(static=True)
    maximum_line_search_steps: int = eqx.field(static=True)
    kkt_regularization: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        mode: InteriorPointMode = "dense-filter",
        linear_policy: LinearSolvePolicy | None = None,
        precision: NonlinearPrecisionPolicy | None = None,
        linear_tolerance: float = 1e-8,
        linear_maximum_steps: int = 200,
        initial_barrier: float = 1e-1,
        barrier_reduction: float = 0.2,
        centering: float = 0.1,
        centering_power: float = 3.0,
        minimum_slack: float = 1e-10,
        minimum_barrier: float = 1e-10,
        fraction_to_boundary: float = 0.995,
        kkt_regularization: float = 1e-8,
        active_tolerance: float = 1e-7,
        sufficient_decrease: float = 1e-4,
        line_search_contraction: float = 0.5,
        filter_margin: float = 1e-4,
        maximum_line_search_steps: int = 24,
        maximum_restoration_steps: int = 20,
        max_dense_dimension: int = 512,
        require_feasible_start: bool = True,
    ):
        if mode == "dense-filter":
            implementation: AbstractMinimizationMethod = _DenseFilterInteriorPoint(
                fraction_to_boundary=fraction_to_boundary,
                minimum_barrier=minimum_barrier,
                filter_margin=filter_margin,
                maximum_line_search_steps=maximum_line_search_steps,
                maximum_restoration_steps=maximum_restoration_steps,
                max_dense_dimension=max_dense_dimension,
                linear=linear_policy,
                precision=precision,
            )
        elif mode == "matrix-free-centered":
            implementation = _MatrixFreeCenteredInteriorPoint(
                linear_policy=linear_policy,
                linear_tolerance=linear_tolerance,
                linear_maximum_steps=linear_maximum_steps,
                initial_barrier=initial_barrier,
                barrier_reduction=barrier_reduction,
                centering=centering,
                minimum_slack=minimum_slack,
                fraction_to_boundary=fraction_to_boundary,
                kkt_regularization=kkt_regularization,
                active_tolerance=active_tolerance,
                sufficient_decrease=sufficient_decrease,
                line_search_contraction=line_search_contraction,
                maximum_line_search_steps=maximum_line_search_steps,
                maximum_restoration_steps=maximum_restoration_steps,
            )
        elif mode == "matrix-free-predictor-corrector":
            implementation = _MatrixFreePredictorCorrector(
                centering_power=centering_power,
                require_feasible_start=require_feasible_start,
                linear_policy=linear_policy,
                linear_tolerance=linear_tolerance,
                linear_maximum_steps=linear_maximum_steps,
                initial_barrier=initial_barrier,
                barrier_reduction=barrier_reduction,
                centering=centering,
                minimum_slack=minimum_slack,
                fraction_to_boundary=fraction_to_boundary,
                kkt_regularization=kkt_regularization,
                active_tolerance=active_tolerance,
                sufficient_decrease=sufficient_decrease,
                line_search_contraction=line_search_contraction,
                maximum_line_search_steps=maximum_line_search_steps,
                maximum_restoration_steps=maximum_restoration_steps,
            )
        elif mode == "sparse-augmented":
            implementation = _DenseFilterInteriorPoint(
                fraction_to_boundary=fraction_to_boundary,
                minimum_barrier=minimum_barrier,
                filter_margin=filter_margin,
                maximum_line_search_steps=maximum_line_search_steps,
                maximum_restoration_steps=maximum_restoration_steps,
                max_dense_dimension=max_dense_dimension,
                precision=precision,
            )
        else:
            raise ValueError(f"Unknown interior-point mode {mode!r}.")
        self.mode = mode
        self.implementation = implementation
        self.structured_linear_policy = (
            LinearSolvePolicy(DenseLU()) if linear_policy is None else linear_policy
        )
        self.fraction_to_boundary = float(fraction_to_boundary)
        self.sufficient_decrease = float(sufficient_decrease)
        self.maximum_line_search_steps = int(maximum_line_search_steps)
        self.kkt_regularization = float(kkt_regularization)

    @property
    def method_id(self) -> str:
        return f"primal-dual-interior-point/{self.mode}"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        implementation = self.implementation.capabilities
        return OptimizationCapabilities(
            scalar_objective=True,
            residual_objective=False,
            matrix_free=implementation.matrix_free,
            prepared_refresh=True,
            implicit_differentiation=True,
        )

    @property
    def structured_capabilities(self) -> StructuredNonlinearCapabilities:
        sparse = self.mode == "sparse-augmented"
        transformed = self.mode.startswith("matrix-free")
        return StructuredNonlinearCapabilities(
            exact_sparse_jacobian=sparse,
            exact_sparse_hessian=sparse,
            limited_memory_hessian=False,
            portable_warm_start=True,
            numeric_refresh=True,
            jit=transformed,
            ordinary_batch=transformed,
            pooled_batch=sparse,
            implicit_differentiation=True,
            device_execution=(
                transformed
                or (
                    isinstance(self.structured_linear_policy.method, SparseLDLT)
                    and sparse_provider_capabilities(
                        "spineax-cudss"
                    ).reliable_zero_inertia
                )
            ),
        )

    def solve(
        self,
        problem: MinimizationProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> MinimizationResult:
        if self.mode == "sparse-augmented":
            raise ValueError(
                "sparse-augmented execution requires a PreparedStructuredNonlinearProgram; "
                "use solve_structured_nonlinear or a domain compiler."
            )
        result = self.implementation.solve(
            problem,
            initial_parameters,
            termination=termination,
            args=args,
        )
        provenance = OptimizationProvenance(
            problem_id=result.provenance.problem_id,
            method=self.method_id,
            backend=result.provenance.backend,
            backend_method=result.provenance.backend_method,
            globalization=result.provenance.globalization,
            matrix_free=result.provenance.matrix_free,
            implicit_differentiation=True,
            notes=result.provenance.notes,
            precision_policy_id=result.provenance.precision_policy_id,
        )
        return eqx.tree_at(lambda value: value.provenance, result, provenance)

    def solve_structured(
        self,
        prepared: PreparedStructuredNonlinearProgram,
        initial_coordinates: Any,
        /,
        *,
        termination: OptimizationTermination,
        warm_start: StructuredNonlinearWarmStart | None,
    ) -> StructuredNonlinearResult:
        if not isinstance(prepared, PreparedStructuredNonlinearProgram):
            raise TypeError("prepared must be a PreparedStructuredNonlinearProgram.")
        if self.mode == "sparse-augmented":
            return solve_sparse_structured_ipm(
                prepared,
                initial_coordinates,
                termination=termination,
                warm_start=warm_start,
                linear_policy=self.structured_linear_policy,
                method_id=self.method_id,
                fraction_to_boundary=self.fraction_to_boundary,
                sufficient_decrease=self.sufficient_decrease,
                maximum_line_search_steps=self.maximum_line_search_steps,
                regularization=self.kkt_regularization,
            )
        program = prepared.program
        constraints = (
            (
                NonlinearConstraint(
                    lambda coordinates, args: program.constraints(coordinates, args),
                    lower=prepared.constraint_lower,
                    upper=prepared.constraint_upper,
                    constraint_id=f"{program.program_id}:structured-constraints",
                ),
            )
            if program.num_constraints
            else ()
        )
        problem = MinimizationProblem(
            lambda coordinates, args: program.objective(coordinates, args),
            bounds=Bounds(prepared.variable_lower, prepared.variable_upper),
            constraints=constraints,
            problem_id=program.program_id,
        )
        initial = prepared.validate_coordinates(
            initial_coordinates if warm_start is None else warm_start.primal
        )
        underlying = self.implementation.solve(
            problem,
            initial,
            termination=termination,
            args=prepared.args,
        )
        (
            constraint_multipliers,
            lower_bound_multipliers,
            upper_bound_multipliers,
        ) = _canonical_structured_multipliers(
            prepared,
            underlying.certificate,
        )
        structured_certificate = prepared.certificate(
            underlying.parameters,
            constraint_multipliers,
            lower_bound_multipliers,
            upper_bound_multipliers,
            active_tolerance=float(jnp.sqrt(termination.absolute_optimality)),
        )
        evaluation = prepared.evaluate(underlying.parameters)
        stationarity = _maximum(jnp.asarray(structured_certificate.stationarity_residual))
        optimality = jnp.maximum(
            stationarity,
            jnp.maximum(
                structured_certificate.primal_feasibility,
                jnp.maximum(
                    structured_certificate.dual_feasibility,
                    structured_certificate.complementarity,
                ),
            ),
        )
        certified = evaluation.finite & (optimality <= termination.absolute_optimality)
        status = jnp.where(
            underlying.successful & ~certified,
            int(OptimizationStatus.CERTIFICATION_FAILED),
            underlying.status,
        ).astype(jnp.int32)
        provenance = OptimizationProvenance(
            problem_id=program.program_id,
            method=self.method_id,
            backend=underlying.provenance.backend,
            backend_method=underlying.provenance.backend_method,
            globalization=underlying.provenance.globalization,
            matrix_free=underlying.provenance.matrix_free,
            implicit_differentiation=True,
            notes=underlying.provenance.notes,
            precision_policy_id=underlying.provenance.precision_policy_id,
        )
        optimization = MinimizationResult(
            underlying.parameters,
            evaluation.objective,
            underlying.auxiliary,
            status,
            _updated_diagnostics(
                underlying.diagnostics,
                structured_certificate,
                optimality,
            ),
            provenance,
            certificate=structured_certificate,
            optimality_certificate=underlying.optimality_certificate,
            method_evidence=underlying.method_evidence,
            precision_evidence=underlying.precision_evidence,
        )
        structured_warm_start = prepared.warm_start(
            underlying.parameters,
            constraint_multipliers,
            lower_bound_multipliers,
            upper_bound_multipliers,
        )
        return StructuredNonlinearResult(
            optimization,
            structured_warm_start,
            _structured_work(underlying, self.mode),
            numeric_version=prepared.numeric_version,
            structure_id=prepared.structure_id,
            method_id=self.method_id,
        )


__all__ = [
    "InteriorPointMode",
    "PrimalDualInteriorPoint",
]
