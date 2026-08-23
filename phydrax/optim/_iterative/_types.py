#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..._bounds import _static_bound_metadata, Bounds
from ..._strict import StrictModule
from ..._tree_math import (  # noqa: F401
    tree_add_scaled as _tree_add_scaled,
    tree_all as _tree_all,
    tree_allfinite as _tree_allfinite,
    tree_inner as _tree_inner,
    tree_negative as _tree_negative,
    tree_norm as _tree_norm,
    tree_scale as _tree_scale,
    tree_where as _tree_where,
    validate_real_inexact_tree as _validate_real_inexact_tree,
)


class OptimizationStatus(IntEnum):
    """Portable status for iterative nonlinear optimization."""

    SUCCESS = 0
    ITERATING = 1
    MAXIMUM_STEPS_REACHED = 2
    MAXIMUM_EVALUATIONS_REACHED = 3
    STAGNATION = 4
    LINE_SEARCH_FAILED = 5
    TRUST_REGION_FAILED = 6
    LINEAR_SOLVE_FAILED = 7
    NONFINITE_INPUT = 8
    NONFINITE_EVALUATION = 9
    INVALID_DIRECTION = 10
    BACKEND_FAILED = 11
    DIVERGENCE = 12
    INFEASIBLE = 13
    CONSTRAINT_QUALIFICATION_FAILED = 14
    RESTORATION_FAILED = 15
    CERTIFICATION_FAILED = 16


_STATUS_MESSAGES = {
    OptimizationStatus.SUCCESS: "success",
    OptimizationStatus.ITERATING: "iteration remains active",
    OptimizationStatus.MAXIMUM_STEPS_REACHED: "maximum steps reached",
    OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED: "maximum evaluations reached",
    OptimizationStatus.STAGNATION: "iteration stagnated before satisfying optimality",
    OptimizationStatus.LINE_SEARCH_FAILED: "line search failed to accept a trial point",
    OptimizationStatus.TRUST_REGION_FAILED: (
        "trust-region damping failed to accept a trial point"
    ),
    OptimizationStatus.LINEAR_SOLVE_FAILED: "linearized subproblem failed",
    OptimizationStatus.NONFINITE_INPUT: "initial parameters contain non-finite values",
    OptimizationStatus.NONFINITE_EVALUATION: (
        "objective, residual, or derivative evaluation was non-finite"
    ),
    OptimizationStatus.INVALID_DIRECTION: "method produced a non-descent direction",
    OptimizationStatus.BACKEND_FAILED: "external optimization backend failed",
    OptimizationStatus.DIVERGENCE: "nonlinear iteration diverged",
    OptimizationStatus.INFEASIBLE: "problem or initial state is infeasible",
    OptimizationStatus.CONSTRAINT_QUALIFICATION_FAILED: (
        "constraint qualification failed"
    ),
    OptimizationStatus.RESTORATION_FAILED: ("constraint feasibility restoration failed"),
    OptimizationStatus.CERTIFICATION_FAILED: (
        "independent final certificate did not pass"
    ),
}


def optimization_status_message(status: int | OptimizationStatus, /) -> str:
    return _STATUS_MESSAGES[OptimizationStatus(int(status))]


class OptimizationTermination(StrictModule):
    """Scale-aware termination policy for one nonlinear solve."""

    absolute_optimality: float = eqx.field(static=True)
    relative_optimality: float = eqx.field(static=True)
    absolute_step: float = eqx.field(static=True)
    relative_step: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    maximum_evaluations: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        absolute_optimality: float = 1e-8,
        relative_optimality: float = 1e-8,
        absolute_step: float = 1e-12,
        relative_step: float = 1e-10,
        maximum_steps: int = 256,
        maximum_evaluations: int | None = None,
    ):
        tolerances = (
            float(absolute_optimality),
            float(relative_optimality),
            float(absolute_step),
            float(relative_step),
        )
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Optimization tolerances must be finite and non-negative.")
        steps = int(maximum_steps)
        evaluations = None if maximum_evaluations is None else int(maximum_evaluations)
        if steps < 1:
            raise ValueError("maximum_steps must be positive.")
        if evaluations is not None and evaluations < 1:
            raise ValueError("maximum_evaluations must be positive or None.")
        (
            self.absolute_optimality,
            self.relative_optimality,
            self.absolute_step,
            self.relative_step,
        ) = tolerances
        self.maximum_steps = steps
        self.maximum_evaluations = evaluations

    def optimality_threshold(self, initial_optimality: Any, /) -> Array:
        initial = jnp.asarray(initial_optimality)
        return self.absolute_optimality + self.relative_optimality * initial

    def step_threshold(self, parameter_norm: Any, /) -> Array:
        norm = jnp.asarray(parameter_norm)
        return self.absolute_step + self.relative_step * norm


class OptimizationCapabilities(StrictModule):
    """Static method capabilities validated before numerical execution."""

    scalar_objective: bool = eqx.field(static=True)
    residual_objective: bool = eqx.field(static=True)
    matrix_free: bool = eqx.field(static=True)
    prepared_refresh: bool = eqx.field(static=True)
    implicit_differentiation: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        scalar_objective: bool,
        residual_objective: bool,
        matrix_free: bool,
        prepared_refresh: bool,
        implicit_differentiation: bool,
    ):
        self.scalar_objective = bool(scalar_objective)
        self.residual_objective = bool(residual_objective)
        self.matrix_free = bool(matrix_free)
        self.prepared_refresh = bool(prepared_refresh)
        self.implicit_differentiation = bool(implicit_differentiation)


class MinimizationProblem(StrictModule):
    """Scalar objective with explicit dynamic argument and auxiliary semantics."""

    objective: Callable[[PyTree[Any], Any], Any]
    bounds: Bounds | None
    constraints: tuple["NonlinearConstraint", ...]
    has_aux: bool = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        objective: Callable[[PyTree[Any], Any], Any],
        /,
        *,
        has_aux: bool = False,
        bounds: Bounds | None = None,
        constraints: Sequence["NonlinearConstraint"] = (),
        problem_id: str = "callable-minimization",
    ):
        if not callable(objective):
            raise TypeError("objective must be callable.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        if bounds is not None and not isinstance(bounds, Bounds):
            raise TypeError("bounds must be a Bounds or None.")
        constraints_ = tuple(constraints)
        if any(
            not isinstance(constraint, NonlinearConstraint) for constraint in constraints_
        ):
            raise TypeError("constraints must contain NonlinearConstraint values.")
        self.objective = objective
        self.has_aux = bool(has_aux)
        self.problem_id = identifier
        self.bounds = bounds
        self.constraints = constraints_

    def value(self, parameters: PyTree[Any], args: Any = None, /) -> tuple[Array, Any]:
        output = self.objective(parameters, args)
        if self.has_aux:
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError(
                    "An objective with has_aux=True must return (value, auxiliary)."
                )
            raw_value, auxiliary = output
        else:
            raw_value, auxiliary = output, None
        value = jnp.asarray(raw_value)
        if value.shape != () or not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("A minimization objective must return one real scalar array.")
        return value, auxiliary

    def value_and_gradient(
        self,
        parameters: PyTree[Any],
        args: Any = None,
        /,
    ) -> tuple[tuple[Array, Any], PyTree[Array]]:
        def value_with_aux(candidate):
            return self.value(candidate, args)

        return eqx.filter_value_and_grad(value_with_aux, has_aux=True)(parameters)


class NonlinearLeastSquaresProblem(StrictModule):
    """Residual-valued nonlinear least-squares problem."""

    residual: Callable[[PyTree[Any], Any], Any]
    bounds: Bounds | None
    has_aux: bool = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        residual: Callable[[PyTree[Any], Any], Any],
        /,
        *,
        has_aux: bool = False,
        bounds: Bounds | None = None,
        problem_id: str = "nonlinear-least-squares",
    ):
        if not callable(residual):
            raise TypeError("residual must be callable.")
        if bounds is not None and not isinstance(bounds, Bounds):
            raise TypeError("bounds must be Bounds or None.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.residual = residual
        self.bounds = bounds
        self.has_aux = bool(has_aux)
        self.problem_id = identifier

    def value(
        self,
        parameters: PyTree[Any],
        args: Any = None,
        /,
    ) -> tuple[PyTree[Array], Any]:
        output = self.residual(parameters, args)
        if self.has_aux:
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError(
                    "A residual with has_aux=True must return (residual, auxiliary)."
                )
            residual, auxiliary = output
        else:
            residual, auxiliary = output, None
        return _validate_real_inexact_tree(residual, name="residual"), auxiliary


class NonlinearConstraint(StrictModule):
    """Bound-form nonlinear constraint ``lower <= function(x, args) <= upper``."""

    function: Callable[[PyTree[Any], Any], PyTree[Any]]
    lower: Any
    upper: Any
    constraint_id: str = eqx.field(static=True)
    _lower_metadata: tuple[tuple[tuple[int, ...], str, tuple[Any, ...]], ...] | None = (
        eqx.field(static=True, repr=False)
    )
    _upper_metadata: tuple[tuple[tuple[int, ...], str, tuple[Any, ...]], ...] | None = (
        eqx.field(static=True, repr=False)
    )

    def __init__(
        self,
        function: Callable[[PyTree[Any], Any], PyTree[Any]],
        /,
        *,
        lower: Any = -jnp.inf,
        upper: Any = jnp.inf,
        constraint_id: str = "nonlinear-constraint",
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        identifier = str(constraint_id)
        if not identifier:
            raise ValueError("constraint_id must be non-empty.")
        self.function = function
        self.lower = lower
        self.upper = upper
        self.constraint_id = identifier
        self._lower_metadata = _static_bound_metadata(lower)
        self._upper_metadata = _static_bound_metadata(upper)

    def value(
        self,
        parameters: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        return _validate_real_inexact_tree(
            self.function(parameters, args),
            name="constraint value",
        )

    def bounds(
        self,
        value: PyTree[Any],
        /,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        value_structure = jax.tree.structure(value)

        def broadcast(bound: Any, *, name: str) -> PyTree[Array]:
            if jax.tree.structure(bound) == value_structure:
                return jax.tree.map(
                    lambda bound_leaf, value_leaf: jnp.broadcast_to(
                        jnp.asarray(bound_leaf, dtype=value_leaf.dtype),
                        value_leaf.shape,
                    ),
                    bound,
                    value,
                )
            bound_array = jnp.asarray(bound)
            if bound_array.shape != ():
                raise ValueError(
                    f"{name} must be scalar or have the constraint value PyTree structure."
                )
            return jax.tree.map(
                lambda leaf: jnp.broadcast_to(
                    bound_array.astype(leaf.dtype),
                    leaf.shape,
                ),
                value,
            )

        lower = broadcast(self.lower, name="lower")
        upper = broadcast(self.upper, name="upper")
        valid = jax.tree.reduce(
            lambda current, pair: current & jnp.all(pair[0] <= pair[1]),
            jax.tree.map(lambda lo, hi: (lo, hi), lower, upper),
            initializer=jnp.asarray(True),
            is_leaf=lambda item: isinstance(item, tuple) and len(item) == 2,
        )
        lower = jax.tree.map(
            lambda leaf: eqx.error_if(
                leaf,
                ~valid,
                "Constraint lower bounds must not exceed upper bounds.",
            ),
            lower,
        )
        return lower, upper


class OptimizationDiagnostics(StrictModule):
    """JAX-compatible evidence from one nonlinear optimization run."""

    iterations: Array
    accepted_steps: Array
    rejected_steps: Array
    objective_evaluations: Array
    gradient_evaluations: Array
    residual_evaluations: Array
    jvp_evaluations: Array
    vjp_evaluations: Array
    hvp_evaluations: Array
    jacobian_evaluations: Array
    constraint_evaluations: Array
    linear_solves: Array
    setup_refreshes: Array
    numeric_refreshes: Array
    linear_iterations: Array
    globalization_evaluations: Array
    initial_optimality_norm: Array
    final_optimality_norm: Array
    final_step_norm: Array
    accepted_step_size: Array
    damping: Array
    reduction_ratio: Array
    direction_fallbacks: Array
    primal_feasibility: Array
    dual_feasibility: Array
    complementarity: Array
    active_constraints: Array
    counts_complete: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        iterations: Any = 0,
        accepted_steps: Any = 0,
        rejected_steps: Any = 0,
        objective_evaluations: Any = 0,
        gradient_evaluations: Any = 0,
        residual_evaluations: Any = 0,
        jvp_evaluations: Any = 0,
        vjp_evaluations: Any = 0,
        hvp_evaluations: Any = 0,
        jacobian_evaluations: Any = 0,
        constraint_evaluations: Any = 0,
        linear_solves: Any = 0,
        setup_refreshes: Any = 0,
        numeric_refreshes: Any = 0,
        linear_iterations: Any = 0,
        globalization_evaluations: Any = 0,
        initial_optimality_norm: Any = jnp.nan,
        final_optimality_norm: Any = jnp.nan,
        final_step_norm: Any = 0.0,
        accepted_step_size: Any = 0.0,
        damping: Any = 0.0,
        reduction_ratio: Any = jnp.nan,
        direction_fallbacks: Any = 0,
        primal_feasibility: Any = 0.0,
        dual_feasibility: Any = jnp.nan,
        complementarity: Any = jnp.nan,
        active_constraints: Any = 0,
        counts_complete: bool = True,
    ):
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.accepted_steps = jnp.asarray(accepted_steps, dtype=jnp.int32)
        self.rejected_steps = jnp.asarray(rejected_steps, dtype=jnp.int32)
        self.objective_evaluations = jnp.asarray(objective_evaluations, dtype=jnp.int32)
        self.gradient_evaluations = jnp.asarray(gradient_evaluations, dtype=jnp.int32)
        self.residual_evaluations = jnp.asarray(residual_evaluations, dtype=jnp.int32)
        self.jvp_evaluations = jnp.asarray(jvp_evaluations, dtype=jnp.int32)
        self.vjp_evaluations = jnp.asarray(vjp_evaluations, dtype=jnp.int32)
        self.hvp_evaluations = jnp.asarray(hvp_evaluations, dtype=jnp.int32)
        self.jacobian_evaluations = jnp.asarray(jacobian_evaluations, dtype=jnp.int32)
        self.constraint_evaluations = jnp.asarray(constraint_evaluations, dtype=jnp.int32)
        self.linear_solves = jnp.asarray(linear_solves, dtype=jnp.int32)
        self.setup_refreshes = jnp.asarray(setup_refreshes, dtype=jnp.int32)
        self.numeric_refreshes = jnp.asarray(numeric_refreshes, dtype=jnp.int32)
        self.linear_iterations = jnp.asarray(linear_iterations, dtype=jnp.int32)
        self.globalization_evaluations = jnp.asarray(
            globalization_evaluations, dtype=jnp.int32
        )
        self.initial_optimality_norm = jnp.asarray(initial_optimality_norm)
        self.final_optimality_norm = jnp.asarray(final_optimality_norm)
        self.final_step_norm = jnp.asarray(final_step_norm)
        self.accepted_step_size = jnp.asarray(accepted_step_size)
        self.damping = jnp.asarray(damping)
        self.reduction_ratio = jnp.asarray(reduction_ratio)
        self.direction_fallbacks = jnp.asarray(direction_fallbacks, dtype=jnp.int32)
        self.primal_feasibility = jnp.asarray(primal_feasibility)
        self.dual_feasibility = jnp.asarray(dual_feasibility)
        self.complementarity = jnp.asarray(complementarity)
        self.active_constraints = jnp.asarray(active_constraints, dtype=jnp.int32)
        self.counts_complete = bool(counts_complete)


class IterativeStepMetrics(StrictModule):
    """Metrics and status from one accepted-point method query."""

    objective: Array
    residual_objective: Array
    scalar_objective: Array
    optimality_norm: Array
    step_norm: Array
    accepted_step_size: Array
    globalization_evaluations: Array
    accepted: Array
    linear_iterations: Array
    linear_status: Array
    forcing: Array
    damping: Array
    reduction_ratio: Array
    direction_fallback: Array
    status: Array

    def __init__(
        self,
        *,
        objective: Any = jnp.nan,
        residual_objective: Any = jnp.nan,
        scalar_objective: Any = jnp.nan,
        optimality_norm: Any = jnp.nan,
        step_norm: Any = 0.0,
        accepted_step_size: Any = 0.0,
        globalization_evaluations: Any = 0,
        accepted: Any = False,
        linear_iterations: Any = 0,
        linear_status: Any = -1,
        forcing: Any = 0.0,
        damping: Any = 0.0,
        reduction_ratio: Any = jnp.nan,
        direction_fallback: Any = False,
        status: Any = OptimizationStatus.ITERATING,
    ):
        objective_ = jnp.asarray(objective)
        scalar_dtype = jnp.result_type(objective_, jnp.float32)
        self.objective = objective_.astype(scalar_dtype)
        self.residual_objective = jnp.asarray(residual_objective, dtype=scalar_dtype)
        self.scalar_objective = jnp.asarray(scalar_objective, dtype=scalar_dtype)
        self.optimality_norm = jnp.asarray(optimality_norm, dtype=scalar_dtype)
        self.step_norm = jnp.asarray(step_norm, dtype=scalar_dtype)
        self.accepted_step_size = jnp.asarray(accepted_step_size, dtype=scalar_dtype)
        self.globalization_evaluations = jnp.asarray(
            globalization_evaluations, dtype=jnp.int32
        )
        self.accepted = jnp.asarray(accepted, dtype=bool)
        self.linear_iterations = jnp.asarray(linear_iterations, dtype=jnp.int32)
        self.linear_status = jnp.asarray(linear_status, dtype=jnp.int32)
        self.forcing = jnp.asarray(forcing, dtype=scalar_dtype)
        self.damping = jnp.asarray(damping, dtype=scalar_dtype)
        self.reduction_ratio = jnp.asarray(reduction_ratio, dtype=scalar_dtype)
        self.direction_fallback = jnp.asarray(direction_fallback, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)


class OptimizationProvenance(StrictModule):
    """Static problem, method, backend, and globalization identity."""

    problem_id: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    backend_method: str = eqx.field(static=True)
    globalization: str = eqx.field(static=True)
    matrix_free: bool = eqx.field(static=True)
    implicit_differentiation: bool = eqx.field(static=True)
    notes: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem_id: str,
        method: str,
        backend: str,
        backend_method: str = "",
        globalization: str,
        matrix_free: bool,
        implicit_differentiation: bool = False,
        notes: str = "",
    ):
        values = tuple(
            str(value) for value in (problem_id, method, backend, globalization)
        )
        if any(not value for value in values):
            raise ValueError("Optimization provenance identifiers must be non-empty.")
        self.problem_id, self.method, self.backend, self.globalization = values
        self.backend_method = str(backend_method)
        self.matrix_free = bool(matrix_free)
        self.implicit_differentiation = bool(implicit_differentiation)
        self.notes = str(notes)


OptimizationCertificateKind: TypeAlias = Literal[
    "unconstrained-stationarity",
    "projected-stationarity",
    "least-squares-normal",
    "composite-gradient-mapping",
    "active-kkt",
    "barrier-kkt",
    "derivative-free-stationarity",
    "global-target",
    "global-bound",
]


class OptimizationCertificate(StrictModule):
    """Independent physical success evidence for one optimization result."""

    tolerance: Array
    optimality_norm: Array
    primal_feasibility: Array
    dual_feasibility: Array
    complementarity: Array
    projected_stationarity: Array
    finite: Array
    regular: Array
    certified: Array
    evaluation_work: Array
    kind: OptimizationCertificateKind = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        kind: OptimizationCertificateKind,
        tolerance: Any,
        optimality_norm: Any,
        primal_feasibility: Any = 0.0,
        dual_feasibility: Any = jnp.nan,
        complementarity: Any = jnp.nan,
        projected_stationarity: Any = jnp.nan,
        finite: Any = True,
        regular: Any = True,
        certified: Any,
        evaluation_work: Any = 0,
        certificate_id: str,
    ):
        identifier = str(certificate_id)
        if not identifier:
            raise ValueError("certificate_id must be non-empty.")
        self.kind = kind
        self.tolerance = jnp.asarray(tolerance)
        self.optimality_norm = jnp.asarray(optimality_norm)
        self.primal_feasibility = jnp.asarray(primal_feasibility)
        self.dual_feasibility = jnp.asarray(dual_feasibility)
        self.complementarity = jnp.asarray(complementarity)
        self.projected_stationarity = jnp.asarray(projected_stationarity)
        self.finite = jnp.asarray(finite, dtype=jnp.bool_)
        self.regular = jnp.asarray(regular, dtype=jnp.bool_)
        self.certified = jnp.asarray(certified, dtype=jnp.bool_)
        self.evaluation_work = jnp.asarray(evaluation_work, dtype=jnp.int32)
        self.certificate_id = identifier


class OptimizationStatusEvidence(StrictModule):
    """Internal-to-public status reconciliation evidence."""

    internal_status: Array
    public_status: Array
    promoted: Array
    demoted: Array
    certificate: OptimizationCertificate
    decision_reason: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        internal_status: Any,
        public_status: Any,
        certificate: OptimizationCertificate,
        promoted: Any,
        demoted: Any,
        decision_reason: str,
    ):
        if not isinstance(certificate, OptimizationCertificate):
            raise TypeError("certificate must be OptimizationCertificate.")
        reason = str(decision_reason)
        if not reason:
            raise ValueError("decision_reason must be non-empty.")
        self.internal_status = jnp.asarray(internal_status, dtype=jnp.int32)
        self.public_status = jnp.asarray(public_status, dtype=jnp.int32)
        self.promoted = jnp.asarray(promoted, dtype=jnp.bool_)
        self.demoted = jnp.asarray(demoted, dtype=jnp.bool_)
        self.certificate = certificate
        self.decision_reason = reason


class ConstrainedOptimalityCertificate(StrictModule):
    """Canonical multipliers, slacks, activity, and KKT residual evidence."""

    equality_multipliers: Array
    inequality_multipliers: Array
    slacks: Array
    active_mask: Array
    stationarity_residual: PyTree[Array]
    primal_feasibility: Array
    dual_feasibility: Array
    complementarity: Array
    equality_sources: tuple[str, ...] = eqx.field(static=True)
    inequality_sources: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        equality_multipliers: Any,
        inequality_multipliers: Any,
        slacks: Any,
        active_mask: Any,
        stationarity_residual: PyTree[Any],
        primal_feasibility: Any,
        dual_feasibility: Any,
        complementarity: Any,
        equality_sources: tuple[str, ...] = (),
        inequality_sources: tuple[str, ...] = (),
    ):
        equality = jnp.asarray(equality_multipliers)
        inequality = jnp.asarray(inequality_multipliers)
        slacks_ = jnp.asarray(slacks)
        active = jnp.asarray(active_mask, dtype=bool)
        if equality.ndim != 1 or inequality.ndim != 1 or slacks_.ndim != 1:
            raise ValueError("Canonical multipliers and slacks must be rank-one arrays.")
        if inequality.shape != slacks_.shape or inequality.shape != active.shape:
            raise ValueError(
                "Inequality multipliers, slacks, and active_mask must have one shape."
            )
        equality_sources_ = tuple(str(source) for source in equality_sources)
        inequality_sources_ = tuple(str(source) for source in inequality_sources)
        if equality_sources_ and len(equality_sources_) != equality.size:
            raise ValueError("equality_sources must identify every equality multiplier.")
        if inequality_sources_ and len(inequality_sources_) != inequality.size:
            raise ValueError(
                "inequality_sources must identify every inequality multiplier."
            )
        self.equality_multipliers = equality
        self.inequality_multipliers = inequality
        self.slacks = slacks_
        self.active_mask = active
        self.stationarity_residual = _validate_real_inexact_tree(
            stationarity_residual,
            name="stationarity_residual",
        )
        self.primal_feasibility = jnp.asarray(primal_feasibility)
        self.dual_feasibility = jnp.asarray(dual_feasibility)
        self.complementarity = jnp.asarray(complementarity)
        self.equality_sources = equality_sources_
        self.inequality_sources = inequality_sources_


class MinimizationResult(StrictModule):
    """Accepted minimizer plus portable status, diagnostics, and provenance."""

    parameters: PyTree[Array]
    objective: Array
    auxiliary: Any
    status: Array
    diagnostics: OptimizationDiagnostics
    provenance: OptimizationProvenance
    certificate: ConstrainedOptimalityCertificate | None
    optimality_certificate: OptimizationCertificate | None
    status_evidence: OptimizationStatusEvidence | None
    method_evidence: Any

    def __init__(
        self,
        parameters: PyTree[Any],
        objective: Any,
        auxiliary: Any,
        status: Any,
        diagnostics: OptimizationDiagnostics,
        provenance: OptimizationProvenance,
        /,
        *,
        certificate: ConstrainedOptimalityCertificate | None = None,
        optimality_certificate: OptimizationCertificate | None = None,
        status_evidence: OptimizationStatusEvidence | None = None,
        method_evidence: Any = None,
    ):
        if not isinstance(diagnostics, OptimizationDiagnostics):
            raise TypeError("diagnostics must be OptimizationDiagnostics.")
        if not isinstance(provenance, OptimizationProvenance):
            raise TypeError("provenance must be OptimizationProvenance.")
        if certificate is not None and not isinstance(
            certificate,
            ConstrainedOptimalityCertificate,
        ):
            raise TypeError(
                "certificate must be a ConstrainedOptimalityCertificate or None."
            )
        if optimality_certificate is not None and not isinstance(
            optimality_certificate,
            OptimizationCertificate,
        ):
            raise TypeError(
                "optimality_certificate must be OptimizationCertificate or None."
            )
        if status_evidence is not None and not isinstance(
            status_evidence,
            OptimizationStatusEvidence,
        ):
            raise TypeError("status_evidence must be OptimizationStatusEvidence or None.")
        self.parameters = parameters
        self.objective = jnp.asarray(objective)
        self.auxiliary = auxiliary
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.diagnostics = diagnostics
        self.provenance = provenance
        self.certificate = certificate
        self.optimality_certificate = optimality_certificate
        self.status_evidence = status_evidence
        self.method_evidence = method_evidence

    @property
    def successful(self) -> Array:
        return self.status == int(OptimizationStatus.SUCCESS)


class LeastSquaresResult(StrictModule):
    """Accepted nonlinear least-squares point and residual evidence."""

    parameters: PyTree[Array]
    residual: PyTree[Array]
    objective: Array
    auxiliary: Any
    status: Array
    diagnostics: OptimizationDiagnostics
    provenance: OptimizationProvenance
    optimality_certificate: OptimizationCertificate | None
    status_evidence: OptimizationStatusEvidence | None
    method_evidence: Any

    def __init__(
        self,
        parameters: PyTree[Any],
        residual: PyTree[Any],
        objective: Any,
        auxiliary: Any,
        status: Any,
        diagnostics: OptimizationDiagnostics,
        provenance: OptimizationProvenance,
        /,
        *,
        optimality_certificate: OptimizationCertificate | None = None,
        status_evidence: OptimizationStatusEvidence | None = None,
        method_evidence: Any = None,
    ):
        if not isinstance(diagnostics, OptimizationDiagnostics):
            raise TypeError("diagnostics must be OptimizationDiagnostics.")
        if not isinstance(provenance, OptimizationProvenance):
            raise TypeError("provenance must be OptimizationProvenance.")
        if optimality_certificate is not None and not isinstance(
            optimality_certificate,
            OptimizationCertificate,
        ):
            raise TypeError(
                "optimality_certificate must be OptimizationCertificate or None."
            )
        if status_evidence is not None and not isinstance(
            status_evidence,
            OptimizationStatusEvidence,
        ):
            raise TypeError("status_evidence must be OptimizationStatusEvidence or None.")
        self.parameters = parameters
        self.residual = residual
        self.objective = jnp.asarray(objective)
        self.auxiliary = auxiliary
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.diagnostics = diagnostics
        self.provenance = provenance
        self.optimality_certificate = optimality_certificate
        self.status_evidence = status_evidence
        self.method_evidence = method_evidence

    @property
    def successful(self) -> Array:
        return self.status == int(OptimizationStatus.SUCCESS)


__all__ = [
    "ConstrainedOptimalityCertificate",
    "OptimizationCertificate",
    "OptimizationCertificateKind",
    "OptimizationStatusEvidence",
    "Bounds",
    "IterativeStepMetrics",
    "LeastSquaresResult",
    "MinimizationProblem",
    "MinimizationResult",
    "NonlinearConstraint",
    "NonlinearLeastSquaresProblem",
    "OptimizationCapabilities",
    "OptimizationDiagnostics",
    "OptimizationProvenance",
    "OptimizationStatus",
    "OptimizationTermination",
    "optimization_status_message",
]
