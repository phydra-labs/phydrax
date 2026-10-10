#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from .._iteration import IterationEvidence, IterationPlan
from .._linear_refresh import prepare_refresh_state
from .._strict import StrictModule
from .._tree_math import (
    tree_add_scaled as _tree_add_scaled,
    tree_allfinite as _tree_allfinite,
    tree_inner as _tree_inner,
    tree_negative as _tree_negative,
    tree_norm as _tree_norm,
    tree_where as _tree_where,
    validate_real_inexact_tree as _validate_real_inexact_tree,
)
from ..linalg import (
    DenseLinearOperator,
    FunctionLinearOperator,
    IdentityLinearOperator,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    MaterializationPolicy,
    materialize,
    MINRES,
    OperatorProperties,
    PreparedLinearization,
    PyTreeSpace,
    solve as solve_linear,
    TolerancePolicy,
)
from ..typing import checked
from ._iteration import attach_terminal_optimization_iteration
from ._iterative._base import AbstractCompositeLeastSquaresMethod
from ._iterative._globalization import armijo_backtracking, ArmijoLineSearch
from ._iterative._types import (
    IterativeStepMetrics,
    OptimizationCapabilities,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)
from ._least_squares import _prepare_residual_model, LeastSquaresState


class CompositeLeastSquaresProblem(StrictModule):
    """Residual squares plus an optional signed scalar objective.

    The represented objective is ``0.5 * ||residual(x, args)||² + scalar(x, args)``.
    The scalar is never converted into an artificial square-root residual.

    ``residual_linearization`` may prepare the exact residual at the supplied
    current parameters and arguments. Its canonical ``PreparedLinearization``
    primal and derivative actions are reused without an extra residual replay.
    The provider owns same-point correctness and must use the canonical
    ``PyTreeSpace(parameters)`` source identity. Provider array leaves are dynamic.
    """

    residual: Callable[[PyTree[Any], Any], PyTree[Any]]
    scalar_objective: Callable[[PyTree[Any], Any], Any] | None
    problem_id: str = eqx.field(static=True)
    residual_linearization: Callable[[PyTree[Any], Any], PreparedLinearization] | None

    def __init__(
        self,
        residual: Callable[[PyTree[Any], Any], PyTree[Any]],
        scalar_objective: Callable[[PyTree[Any], Any], Any] | None = None,
        /,
        *,
        problem_id: str = "composite-least-squares",
        residual_linearization: (
            Callable[[PyTree[Any], Any], PreparedLinearization] | None
        ) = None,
    ) -> None:
        if not callable(residual):
            raise TypeError("residual must be callable.")
        if scalar_objective is not None and not callable(scalar_objective):
            raise TypeError("scalar_objective must be callable or None.")
        if residual_linearization is not None and not callable(residual_linearization):
            raise TypeError("residual_linearization must be callable or None.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.residual = residual
        self.scalar_objective = scalar_objective
        self.problem_id = identifier
        self.residual_linearization = residual_linearization

    def residual_value(
        self,
        parameters: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        return _validate_real_inexact_tree(
            self.residual(parameters, args),
            name="residual",
        )

    def scalar_value(
        self,
        parameters: PyTree[Any],
        args: Any = None,
        /,
    ) -> Array:
        value = (
            jnp.asarray(0.0)
            if self.scalar_objective is None
            else jnp.asarray(self.scalar_objective(parameters, args))
        )
        if value.shape != () or not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("scalar_objective must return one real scalar array.")
        return value

    def objective(
        self,
        parameters: PyTree[Any],
        args: Any = None,
        /,
    ) -> Array:
        residual = self.residual_value(parameters, args)
        return 0.5 * _tree_inner(residual, residual) + self.scalar_value(
            parameters,
            args,
        )


class CompositeLeastSquaresResult(StrictModule):
    """Accepted composite point with separate residual and scalar evidence."""

    parameters: PyTree[Array]
    residual: PyTree[Array]
    residual_objective: Array
    scalar_objective: Array
    objective: Array
    status: Array
    diagnostics: OptimizationDiagnostics
    provenance: OptimizationProvenance
    iteration_evidence: IterationEvidence | None

    @checked
    def __init__(
        self,
        parameters: PyTree[Any],
        residual: PyTree[Any],
        residual_objective: Any,
        scalar_objective: Any,
        objective: Any,
        status: Any,
        diagnostics: OptimizationDiagnostics,
        provenance: OptimizationProvenance,
        /,
        *,
        iteration_evidence: IterationEvidence | None = None,
    ) -> None:
        if iteration_evidence is not None and not isinstance(
            iteration_evidence, IterationEvidence
        ):
            raise TypeError("iteration_evidence must be IterationEvidence or None.")
        self.parameters = parameters
        self.residual = residual
        self.residual_objective = jnp.asarray(residual_objective)
        self.scalar_objective = jnp.asarray(scalar_objective)
        self.objective = jnp.asarray(objective)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.diagnostics = diagnostics
        self.provenance = provenance
        self.iteration_evidence = iteration_evidence

    @property
    def successful(self) -> Array:
        return self.status == int(OptimizationStatus.SUCCESS)


class _CompositeModel(StrictModule):
    residual: PyTree[Array]
    gradient: PyTree[Array]
    residual_objective: Array
    scalar_objective: Array
    objective: Array
    optimality_norm: Array
    residual_model: Any
    scalar_linearized: Any

    def __init__(
        self,
        *,
        residual_model: Any,
        scalar_objective: Any,
        scalar_gradient: PyTree[Any],
        scalar_linearized: Any,
    ) -> None:
        self.residual_model = residual_model
        self.residual = residual_model.residual
        self.gradient = _tree_add_scaled(
            residual_model.gradient,
            scalar_gradient,
            1.0,
        )
        self.residual_objective = residual_model.objective
        self.scalar_objective = jnp.asarray(scalar_objective)
        self.objective = self.residual_objective + self.scalar_objective
        self.optimality_norm = _tree_norm(self.gradient)
        self.scalar_linearized = scalar_linearized

    def curvature_action(
        self,
        vector: PyTree[Any],
        damping: Array,
        /,
    ) -> PyTree[Array]:
        jacobian_vector = self.residual_model.jacobian.mv(vector)
        gauss_newton = self.residual_model.jacobian.adjoint_mv(jacobian_vector)
        _, scalar_hessian_vector = self.scalar_linearized(vector)
        return jax.tree.map(
            lambda residual_part, scalar_part, direction: (
                residual_part + scalar_part + damping * direction
            ),
            gauss_newton,
            scalar_hessian_vector,
            vector,
        )


def _prepare_composite_model(
    problem: CompositeLeastSquaresProblem,
    parameters: PyTree[Any],
    args: Any,
    /,
) -> _CompositeModel:
    if problem.residual_linearization is None:
        residual_model = _prepare_residual_model(
            lambda candidate: problem.residual_value(candidate, args),
            parameters,
        )
    else:
        linearization = problem.residual_linearization(parameters, args)
        if not isinstance(linearization, PreparedLinearization):
            raise TypeError("residual_linearization must return PreparedLinearization.")
        residual_model = _prepare_residual_model(linearization, parameters)
    (scalar_value, scalar_gradient), scalar_linearized = jax.linearize(
        jax.value_and_grad(lambda candidate: problem.scalar_value(candidate, args)),
        parameters,
    )
    return _CompositeModel(
        residual_model=residual_model,
        scalar_objective=scalar_value,
        scalar_gradient=scalar_gradient,
        scalar_linearized=scalar_linearized,
    )


def _default_composite_linear_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        MINRES(),
        tolerance=TolerancePolicy(relative=1e-6, absolute=1e-10),
    )


def _usable_linear_status(status: Any, /) -> Array:
    status_ = jnp.asarray(status, dtype=jnp.int32)
    return (
        (status_ == int(LinearSolveStatus.SUCCESS))
        | (status_ == int(LinearSolveStatus.MAXIMUM_STEPS_REACHED))
        | (status_ == int(LinearSolveStatus.STAGNATION))
        | (status_ == int(LinearSolveStatus.CONDITION_LIMIT_REACHED))
    )


# (trial, damping, found, direction, iterations, matvecs, status, refresh state)
_TrialCarry: TypeAlias = tuple[
    Array, Array, Array, PyTree[Array], Array, Array, Array, PyTree[Any]
]


def _curvature_system(
    space: PyTreeSpace,
    curvature: Callable[[PyTree[Any]], PyTree[Any]] | Array,
    /,
) -> LinearSystem:
    properties = OperatorProperties(
        self_adjoint=True,
        evidence={"self_adjoint": "asserted"},
    )
    if callable(curvature):
        operator = FunctionLinearOperator(
            curvature,
            source=space,
            target=space,
            transpose_action=curvature,
            properties=properties,
            operator_id="composite-generalized-gauss-newton",
        )
    else:
        operator = DenseLinearOperator(
            curvature,
            source=space,
            target=space,
            properties=properties,
            operator_id="composite-generalized-gauss-newton",
        )
    return LinearSystem(operator)


class GeneralizedGaussNewton(AbstractCompositeLeastSquaresMethod):
    """Gauss–Newton plus exact signed-scalar curvature, matrix-free by default."""

    linear_policy: LinearSolvePolicy
    line_search: ArmijoLineSearch
    initial_damping: float = eqx.field(static=True)
    minimum_damping: float = eqx.field(static=True)
    maximum_damping: float = eqx.field(static=True)
    damping_increase: float = eqx.field(static=True)
    damping_decrease: float = eqx.field(static=True)
    maximum_trials: int = eqx.field(static=True)
    curvature_materialization: MaterializationPolicy | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        linear_policy: LinearSolvePolicy | None = None,
        line_search: ArmijoLineSearch | None = None,
        initial_damping: float = 1e-6,
        minimum_damping: float = 1e-12,
        maximum_damping: float = 1e12,
        damping_increase: float = 10.0,
        damping_decrease: float = 0.2,
        maximum_trials: int = 6,
        curvature_materialization: MaterializationPolicy | None = None,
    ) -> None:
        policy = (
            _default_composite_linear_policy() if linear_policy is None else linear_policy
        )
        search = ArmijoLineSearch() if line_search is None else line_search
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
        if not isinstance(search, ArmijoLineSearch):
            raise TypeError("line_search must be an ArmijoLineSearch or None.")
        if curvature_materialization is not None and not isinstance(
            curvature_materialization, MaterializationPolicy
        ):
            raise TypeError(
                "curvature_materialization must be a MaterializationPolicy or None."
            )
        damping = (
            float(initial_damping),
            float(minimum_damping),
            float(maximum_damping),
        )
        if any(not isfinite(value) or value <= 0.0 for value in damping):
            raise ValueError("Damping values must be finite and positive.")
        if not damping[1] <= damping[0] <= damping[2]:
            raise ValueError("initial_damping must lie within the damping bounds.")
        increase = float(damping_increase)
        decrease = float(damping_decrease)
        if not isfinite(increase) or increase <= 1.0:
            raise ValueError("damping_increase must be finite and greater than one.")
        if not isfinite(decrease) or not 0.0 < decrease < 1.0:
            raise ValueError("damping_decrease must lie strictly between zero and one.")
        trials = int(maximum_trials)
        if trials < 1:
            raise ValueError("maximum_trials must be positive.")
        self.linear_policy = policy
        self.line_search = search
        self.initial_damping, self.minimum_damping, self.maximum_damping = damping
        self.damping_increase = increase
        self.damping_decrease = decrease
        self.maximum_trials = trials
        self.curvature_materialization = curvature_materialization

    @property
    def method_id(self) -> str:
        return (
            "generalized-gauss-newton"
            if self.curvature_materialization is None
            else "generalized-gauss-newton-materialized-curvature"
        )

    @property
    def globalization_id(self) -> str:
        return "damped-armijo"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return OptimizationCapabilities(
            scalar_objective=True,
            residual_objective=True,
            matrix_free=self.curvature_materialization is None,
            prepared_refresh=True,
            implicit_differentiation=False,
        )

    def init(self, parameters: PyTree[Any], /) -> LeastSquaresState:
        parameters = _validate_real_inexact_tree(parameters, name="parameters")
        space = PyTreeSpace(parameters)
        if self.curvature_materialization is None:
            system = _curvature_system(space, lambda vector: vector)
        else:
            matrix = materialize(
                IdentityLinearOperator(space), self.curvature_materialization
            )
            system = _curvature_system(space, matrix)
        _, refresh_state = prepare_refresh_state(
            system,
            self.linear_policy,
        )
        metric_nan = jnp.asarray(jnp.nan, dtype=_tree_norm(parameters).dtype)
        damping = jnp.asarray(self.initial_damping, dtype=metric_nan.dtype)
        return LeastSquaresState(
            initial_optimality_norm=metric_nan,
            damping=damping,
            setup_refreshes=1,
            numeric_refreshes=1,
            linear_refresh_state=refresh_state,
            metrics=IterativeStepMetrics(objective=metric_nan),
        )

    @checked
    def prepare_state(
        self,
        problem: CompositeLeastSquaresProblem,
        parameters: PyTree[Any],
        /,
        *,
        args: Any,
    ) -> LeastSquaresState:
        return self.init(parameters)

    @checked
    def step(
        self,
        problem: CompositeLeastSquaresProblem,
        parameters: PyTree[Any],
        state: LeastSquaresState,
        /,
        *,
        termination: OptimizationTermination | None,
        args: Any,
    ) -> tuple[PyTree[Any], LeastSquaresState, Array]:
        # Keep the traced accept/terminal state machine together: its branches
        # share one prepared model and the exact dynamic refresh-state layout.
        # Moving preparation across either guard would execute failed/converged
        # curvature actions or change the compiler and work-evidence boundary.
        if state.linear_refresh_state is None:
            raise ValueError(
                "GeneralizedGaussNewton state is missing linear refresh state."
            )
        _, static_state = eqx.partition(state, eqx.is_array)
        model = _prepare_composite_model(problem, parameters, args)
        initial_optimality = jnp.where(
            state.iteration == 0,
            model.optimality_norm,
            state.initial_optimality_norm,
        )
        finite_model = (
            jnp.isfinite(model.objective)
            & jnp.isfinite(model.optimality_norm)
            & _tree_allfinite(model.residual)
            & _tree_allfinite(model.gradient)
        )
        converged = (
            jnp.asarray(False)
            if termination is None
            else model.optimality_norm
            <= termination.optimality_threshold(initial_optimality)
        )

        def terminal_step(_: None) -> tuple[PyTree[Any], PyTree[Any], Array]:
            status = jnp.where(
                finite_model,
                int(OptimizationStatus.SUCCESS),
                int(OptimizationStatus.NONFINITE_EVALUATION),
            )
            metrics = IterativeStepMetrics(
                objective=model.objective,
                residual_objective=model.residual_objective,
                scalar_objective=model.scalar_objective,
                optimality_norm=model.optimality_norm,
                damping=state.damping,
                accepted=finite_model,
                status=status,
            )
            updated = LeastSquaresState(
                iteration=state.iteration + 1,
                initial_optimality_norm=initial_optimality,
                damping=state.damping,
                accepted_steps=state.accepted_steps,
                rejected_steps=state.rejected_steps + (~finite_model).astype(jnp.int32),
                residual_evaluations=state.residual_evaluations + 1,
                scalar_evaluations=state.scalar_evaluations + 1,
                scalar_gradient_evaluations=state.scalar_gradient_evaluations + 1,
                scalar_hvp_evaluations=state.scalar_hvp_evaluations,
                jvp_evaluations=state.jvp_evaluations,
                vjp_evaluations=state.vjp_evaluations + 1,
                linear_iterations=state.linear_iterations,
                linear_solves=state.linear_solves,
                direction_fallbacks=state.direction_fallbacks,
                setup_refreshes=state.setup_refreshes,
                numeric_refreshes=state.numeric_refreshes,
                linear_refresh_state=state.linear_refresh_state,
                metrics=metrics,
            )
            dynamic_updated, _ = eqx.partition(updated, eqx.is_array)
            return parameters, dynamic_updated, model.objective

        def composite_step(_: None) -> tuple[PyTree[Any], PyTree[Any], Array]:
            space = PyTreeSpace(parameters)
            if self.curvature_materialization is None:
                curvature_matrix = None
            else:

                def undamped_curvature_action(vector: PyTree[Any]) -> PyTree[Array]:
                    return model.curvature_action(vector, jnp.zeros_like(state.damping))

                curvature_system = _curvature_system(space, undamped_curvature_action)
                # The current-point curvature is prepared once. Trial damping
                # shifts the numerical matrix without retracing model actions.
                curvature_matrix = materialize(
                    curvature_system.operator, self.curvature_materialization
                )

            dynamic_refresh_state, static_refresh_state = eqx.partition(
                state.linear_refresh_state,
                eqx.is_array,
            )

            def trial_condition(carry: _TrialCarry) -> Array:
                trial, _, found, *_ = carry
                return (trial < self.maximum_trials) & (~found)

            def trial_body(carry: _TrialCarry) -> _TrialCarry:
                (
                    trial,
                    damping,
                    _,
                    direction,
                    iterations,
                    matvecs,
                    last_status,
                    dynamic_refresh_state_for_trial,
                ) = carry

                refresh_state_for_trial = eqx.combine(
                    dynamic_refresh_state_for_trial,
                    static_refresh_state,
                )

                if curvature_matrix is None:

                    def curvature_action(vector: PyTree[Any]) -> PyTree[Array]:
                        return model.curvature_action(vector, damping)

                    system = _curvature_system(space, curvature_action)
                else:
                    matrix = curvature_matrix + damping * jnp.eye(
                        space.size, dtype=curvature_matrix.dtype
                    )
                    system = _curvature_system(space, matrix)
                current_prepared, current_refresh_state = refresh_state_for_trial.refresh(
                    system
                )
                linear_result = solve_linear(
                    current_prepared,
                    _tree_negative(model.gradient),
                )
                proposed = linear_result.value
                directional = _tree_inner(model.gradient, proposed)
                usable = (
                    _usable_linear_status(linear_result.status)
                    & _tree_allfinite(proposed)
                    & jnp.isfinite(directional)
                    & (directional < 0.0)
                )
                dynamic_current_refresh_state, _ = eqx.partition(
                    current_refresh_state,
                    eqx.is_array,
                )
                return (
                    trial + 1,
                    jnp.where(
                        usable,
                        damping,
                        jnp.minimum(
                            self.maximum_damping,
                            damping * self.damping_increase,
                        ),
                    ),
                    usable,
                    _tree_where(usable, proposed, direction),
                    iterations
                    + jnp.asarray(
                        linear_result.diagnostics.iterations,
                        dtype=jnp.int32,
                    ).reshape(()),
                    matvecs
                    + jnp.asarray(
                        linear_result.diagnostics.matvec_count,
                        dtype=jnp.int32,
                    ).reshape(()),
                    jnp.asarray(linear_result.status, dtype=jnp.int32),
                    dynamic_current_refresh_state,
                )

            (
                trials,
                damping,
                found,
                proposed_direction,
                linear_iterations,
                matvecs,
                last_linear_status,
                final_refresh_state,
            ) = jax.lax.while_loop(
                trial_condition,
                trial_body,
                (
                    jnp.asarray(0, dtype=jnp.int32),
                    state.damping,
                    jnp.asarray(False),
                    _tree_negative(model.gradient),
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(-1, dtype=jnp.int32),
                    dynamic_refresh_state,
                ),
            )
            final_refresh_state = eqx.combine(
                final_refresh_state,
                static_refresh_state,
            )
            direction = _tree_where(
                found,
                proposed_direction,
                _tree_negative(model.gradient),
            )
            directional = _tree_inner(model.gradient, direction)
            search = armijo_backtracking(
                lambda candidate: problem.objective(candidate, args),
                parameters,
                model.objective,
                direction,
                directional,
                step=_tree_add_scaled,
                contains=_tree_allfinite,
                policy=self.line_search,
            )
            accepted = search.accepted
            step_norm = search.rate * _tree_norm(direction)
            stagnated = (
                jnp.asarray(False)
                if termination is None
                else accepted
                & (step_norm <= termination.step_threshold(_tree_norm(parameters)))
            )
            status = jnp.where(
                stagnated,
                int(OptimizationStatus.STAGNATION),
                jnp.where(
                    accepted,
                    int(OptimizationStatus.ITERATING),
                    jnp.where(
                        search.finite_candidate_seen,
                        int(OptimizationStatus.LINE_SEARCH_FAILED),
                        int(OptimizationStatus.NONFINITE_EVALUATION),
                    ),
                ),
            )
            next_damping = jnp.where(
                accepted & found,
                jnp.maximum(
                    self.minimum_damping,
                    damping * self.damping_decrease,
                ),
                damping,
            )
            curvature_products = (
                matvecs
                if curvature_matrix is None
                else jnp.asarray(space.size, dtype=jnp.int32)
            )
            metrics = IterativeStepMetrics(
                objective=search.value,
                residual_objective=model.residual_objective,
                scalar_objective=model.scalar_objective,
                optimality_norm=model.optimality_norm,
                step_norm=step_norm,
                accepted_step_size=search.rate,
                globalization_evaluations=search.evaluations,
                accepted=accepted,
                linear_iterations=linear_iterations,
                linear_status=last_linear_status,
                damping=next_damping,
                direction_fallback=~found,
                status=status,
            )
            updated = LeastSquaresState(
                iteration=state.iteration + 1,
                initial_optimality_norm=initial_optimality,
                damping=next_damping,
                accepted_steps=state.accepted_steps + accepted.astype(jnp.int32),
                rejected_steps=state.rejected_steps + (~accepted).astype(jnp.int32),
                residual_evaluations=(
                    state.residual_evaluations + 1 + search.evaluations
                ),
                scalar_evaluations=(state.scalar_evaluations + 1 + search.evaluations),
                scalar_gradient_evaluations=(state.scalar_gradient_evaluations + 1),
                scalar_hvp_evaluations=(
                    state.scalar_hvp_evaluations + curvature_products
                ),
                jvp_evaluations=state.jvp_evaluations + curvature_products,
                vjp_evaluations=state.vjp_evaluations + 1 + curvature_products,
                linear_iterations=state.linear_iterations + linear_iterations,
                linear_solves=state.linear_solves + trials,
                setup_refreshes=state.setup_refreshes,
                numeric_refreshes=state.numeric_refreshes + trials,
                linear_refresh_state=final_refresh_state,
                direction_fallbacks=(
                    state.direction_fallbacks + (~found).astype(jnp.int32)
                ),
                metrics=metrics,
            )
            dynamic_updated, _ = eqx.partition(updated, eqx.is_array)
            return search.parameters, dynamic_updated, search.value

        parameters, dynamic_updated, objective = jax.lax.cond(
            (~finite_model) | converged,
            terminal_step,
            composite_step,
            None,
        )
        return (
            parameters,
            eqx.combine(dynamic_updated, static_state),
            objective,
        )

    @checked
    def step_metrics(self, state: LeastSquaresState, /) -> IterativeStepMetrics:
        return state.metrics

    def solve(
        self,
        problem: CompositeLeastSquaresProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> CompositeLeastSquaresResult:
        return _solve_composite(
            self,
            problem,
            initial_parameters,
            termination=termination,
            args=args,
        )


def _iterate_composite(
    method: GeneralizedGaussNewton,
    problem: CompositeLeastSquaresProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
) -> tuple[PyTree[Any], LeastSquaresState, Array]:
    """Prepare iteration state, then execute the native optimization recurrence."""
    if not isinstance(problem, CompositeLeastSquaresProblem):
        raise TypeError("problem must be a CompositeLeastSquaresProblem.")
    if not isinstance(termination, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination.")
    state = method.prepare_state(problem, initial_parameters, args=args)
    return _iterate_prepared_composite(
        method,
        problem,
        initial_parameters,
        state,
        termination=termination,
        args=args,
    )


def _iterate_prepared_composite(
    method: GeneralizedGaussNewton,
    problem: CompositeLeastSquaresProblem,
    initial_parameters: PyTree[Any],
    state: LeastSquaresState,
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
) -> tuple[PyTree[Any], LeastSquaresState, Array]:
    """Execute iterations from one shape-compatible native method state."""
    if not isinstance(problem, CompositeLeastSquaresProblem):
        raise TypeError("problem must be a CompositeLeastSquaresProblem.")
    if not isinstance(state, LeastSquaresState):
        raise TypeError("state must be a LeastSquaresState.")
    if not isinstance(termination, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination.")
    initial = _validate_real_inexact_tree(
        initial_parameters,
        name="initial_parameters",
    )
    state, static_state = eqx.partition(state, eqx.is_array)
    initial_status = jnp.where(
        _tree_allfinite(initial),
        int(OptimizationStatus.ITERATING),
        int(OptimizationStatus.NONFINITE_INPUT),
    ).astype(jnp.int32)

    def condition(carry: tuple[PyTree[Any], PyTree[Any], Array]) -> Array:
        _, current_state, status = carry
        evaluations = jnp.maximum(
            current_state.residual_evaluations,
            current_state.scalar_evaluations,
        )
        within_evaluations = (
            jnp.asarray(True)
            if termination.maximum_evaluations is None
            else evaluations < termination.maximum_evaluations
        )
        return (
            (status == int(OptimizationStatus.ITERATING))
            & (current_state.iteration < termination.maximum_steps)
            & within_evaluations
        )

    def body(
        carry: tuple[PyTree[Any], PyTree[Any], Array],
    ) -> tuple[PyTree[Any], PyTree[Any], Array]:
        parameters, dynamic_state, _ = carry
        current_state = eqx.combine(dynamic_state, static_state)
        parameters, next_state, _ = method.step(
            problem,
            parameters,
            current_state,
            termination=termination,
            args=args,
        )
        status = method.step_metrics(next_state).status
        if termination.maximum_evaluations is not None:
            evaluations = jnp.maximum(
                next_state.residual_evaluations,
                next_state.scalar_evaluations,
            )
            status = jnp.where(
                (status == int(OptimizationStatus.ITERATING))
                & (evaluations >= termination.maximum_evaluations),
                int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
                status,
            )
        dynamic_next_state, _ = eqx.partition(next_state, eqx.is_array)
        return parameters, dynamic_next_state, status

    parameters, state, status = jax.lax.while_loop(
        condition,
        body,
        (initial, state, initial_status),
    )
    state = eqx.combine(state, static_state)
    status = jnp.where(
        status == int(OptimizationStatus.ITERATING),
        int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
        status,
    )
    return parameters, state, status


def _finalize_composite(
    method: GeneralizedGaussNewton,
    problem: CompositeLeastSquaresProblem,
    parameters: PyTree[Any],
    state: LeastSquaresState,
    status: Array,
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
) -> CompositeLeastSquaresResult:
    """Evaluate and package exact final-point status after native iterations."""
    final_model = _prepare_composite_model(problem, parameters, args)
    finite_final = (
        jnp.isfinite(final_model.objective)
        & jnp.isfinite(final_model.optimality_norm)
        & _tree_allfinite(parameters)
        & _tree_allfinite(final_model.residual)
        & _tree_allfinite(final_model.gradient)
    )
    initial_nonfinite = status == int(OptimizationStatus.NONFINITE_INPUT)
    status = jnp.where(
        initial_nonfinite,
        status,
        jnp.where(
            ~finite_final,
            int(OptimizationStatus.NONFINITE_EVALUATION),
            jnp.where(
                final_model.optimality_norm
                <= termination.optimality_threshold(state.initial_optimality_norm),
                int(OptimizationStatus.SUCCESS),
                status,
            ),
        ),
    )
    metrics = method.step_metrics(state)
    diagnostics = OptimizationDiagnostics(
        iterations=state.iteration,
        accepted_steps=state.accepted_steps,
        rejected_steps=state.rejected_steps,
        objective_evaluations=state.scalar_evaluations + 1,
        gradient_evaluations=state.scalar_gradient_evaluations + 1,
        residual_evaluations=state.residual_evaluations + 1,
        jvp_evaluations=state.jvp_evaluations,
        vjp_evaluations=state.vjp_evaluations + 1,
        hvp_evaluations=state.scalar_hvp_evaluations,
        linear_solves=state.linear_solves,
        setup_refreshes=state.setup_refreshes,
        numeric_refreshes=state.numeric_refreshes,
        linear_iterations=state.linear_iterations,
        globalization_evaluations=(state.scalar_evaluations - state.iteration),
        initial_optimality_norm=state.initial_optimality_norm,
        final_optimality_norm=final_model.optimality_norm,
        final_step_norm=metrics.step_norm,
        accepted_step_size=metrics.accepted_step_size,
        damping=state.damping,
        direction_fallbacks=state.direction_fallbacks,
    )
    return CompositeLeastSquaresResult(
        parameters,
        final_model.residual,
        final_model.residual_objective,
        final_model.scalar_objective,
        final_model.objective,
        status,
        diagnostics,
        OptimizationProvenance(
            problem_id=problem.problem_id,
            method=method.method_id,
            backend="phydrax-native",
            globalization=method.globalization_id,
            matrix_free=True,
            notes="Exact scalar Hessian actions augment the Gauss-Newton model.",
        ),
    )


def _solve_composite(
    method: GeneralizedGaussNewton,
    problem: CompositeLeastSquaresProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
) -> CompositeLeastSquaresResult:
    parameters, state, status = _iterate_composite(
        method,
        problem,
        initial_parameters,
        termination=termination,
        args=args,
    )
    return _finalize_composite(
        method,
        problem,
        parameters,
        state,
        status,
        termination=termination,
        args=args,
    )


def composite_least_squares(
    problem: CompositeLeastSquaresProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    method: AbstractCompositeLeastSquaresMethod | None = None,
    termination: OptimizationTermination | None = None,
    args: Any = None,
    iteration: IterationPlan | None = None,
) -> CompositeLeastSquaresResult:
    """Minimize one residual-plus-scalar objective with explicit semantics."""

    method_ = GeneralizedGaussNewton() if method is None else method
    termination_ = OptimizationTermination() if termination is None else termination
    if not isinstance(method_, AbstractCompositeLeastSquaresMethod):
        raise TypeError("method must be an AbstractCompositeLeastSquaresMethod or None.")
    if iteration is not None and not isinstance(iteration, IterationPlan):
        raise TypeError("iteration must be IterationPlan or None.")
    if iteration is not None and iteration.granularity != "terminal":
        raise ValueError(
            "Composite least squares currently supports terminal iteration evidence."
        )
    result = method_.solve(
        problem,
        initial_parameters,
        termination=termination_,
        args=args,
    )
    return attach_terminal_optimization_iteration(
        result,
        iteration,
        method_.method_id,
    )


__all__ = [
    "CompositeLeastSquaresProblem",
    "CompositeLeastSquaresResult",
    "GeneralizedGaussNewton",
    "composite_least_squares",
]
