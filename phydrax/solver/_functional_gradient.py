#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._frozendict import frozendict
from .._iteration import IterationSession
from .._strict import StrictModule
from .._trainable import partition_parameters
from .._training import (
    DelayedTargetPolicy,
    emit_training_signal_stop as _emit_training_signal_stop,
    EvaluationParametersFn,
    ExponentialMovingAverageTargetPolicy,
    resolve_evaluation_parameters,
    tensorboard_every as _tensorboard_every,
    TensorBoardLogger as _TensorBoardLogger,
    TrainingController,
    TrainingIterationKind,
    TrainingProgress,
    TrainingSignalGuard as _TrainingSignalGuard,
)
from .._training_kernel import (
    AbstractKernelUpdateRule,
    KernelUpdateContext,
    OptaxUpdateRule,
    prepare_training_kernel,
    PreparedTrainingKernel,
    run_training_attempt,
    TrainingAttemptOutcome,
    TrainingKernelSpec,
    TrainingKernelState,
)
from .._training_objective import _ObjectiveAccumulator, _ObjectiveContribution
from .._tree_math import tree_negative, tree_where
from ..logging import is_enabled as _logging_enabled
from ..nn.parameters import ParameterSubspace
from ..nn.parameters._low_rank import validate_low_rank_subspace
from ..optim._composite import CompositeLeastSquaresProblem
from ..optim._gradient_composition import (
    conflict_free_gradient,
    ConflictFreeGradientPolicy,
)
from ..optim._iterative import (
    AbstractCompositeLeastSquaresMethod,
    AbstractLeastSquaresMethod,
    AbstractScalarIterativeMethod,
    OptimizationStatus,
    OptimizationTermination,
)
from ..optim._kernel_rules import (
    CompositeLeastSquaresUpdateRule,
    LeastSquaresUpdateRule,
    MirrorUpdateRule,
    NativeMethodState,
    RiemannianLineSearchUpdateRule,
    RiemannianUpdateRule,
    ScalarIterativeUpdateRule,
)
from ..optim._least_squares import LeastSquaresState
from ..optim._mirror_descent import AbstractMirrorOptimizer
from ..optim._riemannian import (
    AbstractRiemannianLineSearchOptimizer,
    AbstractRiemannianOptimizer,
)
from ..optim._scalar import ScalarIterativeState
from ..optim._update_alignment import (
    _alignment_conflicts,
    ConflictFreeUpdatePolicy,
    ConflictFreeUpdateResult,
    ConflictFreeUpdateStatistics,
    project_conflict_free_direction,
)
from ._functional_checkpoint import (
    load_functional_training_checkpoint,
    save_functional_training_checkpoint,
)
from ._functional_kernel import (
    functional_kernel_objective,
    functional_lanes,
    functional_parameter_lane,
    FUNCTIONAL_REJECTION_BUDGET,
    FUNCTIONAL_ROOT_AUTHORITY,
    functional_site_key,
    functional_site_keys,
    functional_training_tree,
    functional_tree_functions,
    resume_functional_kernel_state,
)
from ._functional_objective import (
    evaluate_prepared_objective,
    evaluate_prepared_scalar_remainder,
    prepared_data_metrics,
)
from ._functional_precision import FunctionalPrecisionPolicy
from ._functional_reporting import (
    best_display_value as _best_display_value,
    emit_training_scalars as _emit_training_scalars,
    term_label as _term_label,
    training_scalars as _training_scalars,
    write_tensorboard_scalars as _write_tensorboard_scalars,
)
from ._functional_residual import (
    materialize_prepared_residual_terms,
    prepared_term_residual_vector,
)
from ._functional_run import (
    expand_train_terms as _expanded_train_terms,
    replace_solver_state,
    select_train_terms as _active_train_terms,
    validate_term_sample_size as _train_term_sample_size,
)
from ._functional_surrogate import (
    _functional_ntk_diagnostic_values,
    _functional_ntk_diagnostics,
    prepare_functional_update,
    PreparedFunctionalUpdate,
)
from ._functional_training import (
    FunctionalTrainingPlan,
    FunctionalTrainingState,
)
from ._model_losses import function_model_loss_labels


if TYPE_CHECKING:
    from ._functional_solver import FunctionalSolver


_KERNEL_CONTEXT = "FunctionalSolver.solve"
_LINE_SEARCH_STATE_TYPES = (
    optax.ScaleByBacktrackingLinesearchState,
    optax.ScaleByZoomLinesearchState,
)


@dataclass(frozen=True, slots=True)
class _FunctionalOptimizerRoute:
    line_search: optax.GradientTransformationExtraArgs | None
    standard: optax.GradientTransformation | None
    composite: AbstractCompositeLeastSquaresMethod | None
    iterative: AbstractScalarIterativeMethod | None
    least_squares: AbstractLeastSquaresMethod | None
    mirror: AbstractMirrorOptimizer | None
    riemannian: AbstractRiemannianOptimizer | None
    label: str

    @property
    def native(self) -> AbstractCompositeLeastSquaresMethod | Any:
        """Native least-squares or iterative method that owns its globalization."""
        return self.composite or self.least_squares or self.iterative

    @property
    def riemannian_line_search(self) -> bool:
        return isinstance(self.riemannian, AbstractRiemannianLineSearchOptimizer)

    @property
    def evaluates_candidate(self) -> bool:
        """Whether the route's loss and selection are read at the updated point."""
        return (
            self.native is not None
            or self.line_search is not None
            or self.riemannian_line_search
        )


def _resolve_functional_optimizer(
    optim: Any,
    evaluation_parameters: EvaluationParametersFn | None,
    parameter_paths: tuple[str, ...] | None,
    precision: FunctionalPrecisionPolicy | None,
    /,
) -> _FunctionalOptimizerRoute:
    if isinstance(optim, str):
        raise TypeError(
            "optim must be a Phydrax mirror or Riemannian optimizer, or an Optax transformation, not a string."
        )

    _opt_linesearch: optax.GradientTransformationExtraArgs | None = None
    _opt_standard: optax.GradientTransformation | None = None
    _opt_composite: AbstractCompositeLeastSquaresMethod | None = None
    _opt_iterative: AbstractScalarIterativeMethod | None = None
    _opt_least_squares: AbstractLeastSquaresMethod | None = None

    _opt_mirror: AbstractMirrorOptimizer | None = None
    _opt_riemannian: AbstractRiemannianOptimizer | None = None
    if isinstance(optim, AbstractCompositeLeastSquaresMethod):
        _opt_composite = optim
    elif isinstance(optim, AbstractLeastSquaresMethod):
        _opt_least_squares = optim
    elif isinstance(optim, AbstractScalarIterativeMethod):
        _opt_iterative = optim
    elif isinstance(optim, AbstractMirrorOptimizer):
        if evaluation_parameters is not None:
            raise ValueError(
                "evaluation_parameters is unsupported for mirror optimizers because "
                "an ambient transform need not preserve Legendre support."
            )
        _opt_mirror = optim
    elif isinstance(optim, AbstractRiemannianOptimizer):
        if evaluation_parameters is not None:
            raise ValueError(
                "evaluation_parameters is unsupported for Riemannian optimizers "
                "because an ambient transform need not preserve manifold membership."
            )
        _opt_riemannian = optim
    elif isinstance(optim, optax.GradientTransformationExtraArgs):
        _opt_linesearch = optim
    elif isinstance(optim, optax.GradientTransformation):
        _opt_standard = optim
    else:
        raise TypeError(
            "optim must be a Phydrax least-squares, iterative, mirror, or "
            "Riemannian optimizer, or an Optax transformation."
        )
    if evaluation_parameters is not None and (
        _opt_composite is not None
        or _opt_least_squares is not None
        or _opt_iterative is not None
    ):
        raise ValueError(
            "evaluation_parameters is supported only by Optax transformations: "
            "native least-squares and iterative methods own no evaluation view."
        )
    if parameter_paths is not None and _opt_standard is None and _opt_linesearch is None:
        raise ValueError(
            "Explicit parameter subspaces are supported only by Optax transformations."
        )
    if precision is not None and not isinstance(precision, FunctionalPrecisionPolicy):
        raise TypeError("precision must be a FunctionalPrecisionPolicy or None.")
    optimizer_label = (
        _opt_composite.method_id
        if _opt_composite is not None
        else _opt_least_squares.method_id
        if _opt_least_squares is not None
        else _opt_iterative.method_id
        if _opt_iterative is not None
        else _opt_mirror.optimizer_id
        if _opt_mirror is not None
        else _opt_riemannian.optimizer_id
        if _opt_riemannian is not None
        else "optax"
    )
    return _FunctionalOptimizerRoute(
        _opt_linesearch,
        _opt_standard,
        _opt_composite,
        _opt_iterative,
        _opt_least_squares,
        _opt_mirror,
        _opt_riemannian,
        optimizer_label,
    )


def _with_line_search_route(
    route: _FunctionalOptimizerRoute, parameters: Any, /
) -> _FunctionalOptimizerRoute:
    """Demote an extra-argument Optax transformation without a line search."""
    if route.line_search is None:
        return route
    state = route.line_search.init(parameters)
    leaves = jax.tree.leaves(
        state, is_leaf=lambda value: isinstance(value, _LINE_SEARCH_STATE_TYPES)
    )
    if any(isinstance(value, _LINE_SEARCH_STATE_TYPES) for value in leaves):
        return route
    return replace(route, line_search=None, standard=route.line_search)


# Functional objective over the legacy lanes --------------------------------------


def _reconstruct_functions(trained: Any, held: Any, /) -> Any:
    if isinstance(held, ParameterSubspace):
        return held.reconstruct(trained)
    return eqx.combine(trained, held)


def _surrogate_coordinates(
    trained: Any, held: Any, surrogate_filter: Any, /
) -> tuple[Any, Any]:
    """Surrogate coordinates: the trained lane, or every selected and alias leaf."""
    if surrogate_filter is None:
        return trained, held
    return eqx.partition(_reconstruct_functions(trained, held), surrogate_filter)


def _physical_prepared(prepared: Any, /) -> Any:
    return (
        prepared.physical if isinstance(prepared, PreparedFunctionalUpdate) else prepared
    )


def _precision_context(precision: FunctionalPrecisionPolicy | None, /) -> Any:
    return (
        nullcontext()
        if precision is None
        else jax.default_matmul_precision(precision.matmul_precision)
    )


def _objective_values(
    trained: Any,
    held: Any,
    prepared: Any,
    precision: FunctionalPrecisionPolicy | None,
    surrogate_filter: Any,
    /,
) -> tuple[Array, Array, Array]:
    """Ordered total, flat term values, and gradient components of one objective."""
    if isinstance(prepared, PreparedFunctionalUpdate):
        values = prepared.surrogate_values(
            *_surrogate_coordinates(trained, held, surrogate_filter)
        )
        return values.total, values.flat_values, values.gradient_values
    functions = _reconstruct_functions(trained, held)
    with _precision_context(precision):
        values = evaluate_prepared_objective(prepared, functions)
    return values.total, values.flat_values, values.gradient_values


_compiled_objective_values = eqx.filter_jit(_objective_values)


@final
class FunctionalLossDiagnostics(StrictModule):
    """Per-attempt diagnostics of the FunctionalSolver objective.

    `term_values` are the flat authored term and model-loss values at the
    attempt's parameters; `component_gradients` are the gradients of the
    objective components when a conflict-free update rule consumes them.
    """

    term_values: Array
    component_gradients: tuple[Any, ...] | None


@final
class _FunctionalGradientLoss(StrictModule):
    """The ordered FunctionalSolver total on the attempt's prepared objective."""

    precision: FunctionalPrecisionPolicy | None
    surrogate_filter: Any
    components: bool = eqx.field(static=True)

    def __call__(
        self, trained: Any, held: Any, prepared: Any, keys: Any
    ) -> tuple[Array, FunctionalLossDiagnostics]:
        del keys  # The prepared payload carries every realization of the attempt.
        if not self.components:
            total, flat_values, _ = _objective_values(
                trained, held, prepared, self.precision, self.surrogate_filter
            )
            return total, FunctionalLossDiagnostics(flat_values, None)

        def values(parameters: Any) -> tuple[Array, Array, Array]:
            return _objective_values(
                parameters, held, prepared, self.precision, self.surrogate_filter
            )

        (total, flat_values, component_values), pullback = eqx.filter_vjp(values, trained)
        gradients = tuple(
            pullback(
                (
                    jnp.zeros_like(total),
                    jnp.zeros_like(flat_values),
                    jnp.zeros_like(component_values).at[index].set(1),
                )
            )[0]
            for index in range(component_values.shape[0])
        )
        return total, FunctionalLossDiagnostics(flat_values, gradients)


# Update rules ----------------------------------------------------------------------


@final
class ConflictFreeOptaxState(StrictModule):
    """Optax state plus conflict-free alignment evidence of the last update."""

    optimizer_state: Any
    statistics: ConflictFreeUpdateStatistics | None
    result: ConflictFreeUpdateResult | None


def _lane_gradient(lane: Any, trained: Any, /) -> Any:
    """Place a legacy trained-lane gradient into the kernel PARAMETER lane."""
    if functional_parameter_lane(lane) is lane:
        return trained
    return eqx.tree_at(lambda value: value.selected, lane, trained)


@final
class _ConflictFreeOptaxRule(AbstractKernelUpdateRule):
    """Standard Optax with conflict-free gradient composition or update alignment.

    The objective's component gradients (computed under the kernel's admission
    mask) replace the gradient by their conflict-free composition, or the Optax
    proposal is projected against them. An unsuccessful composition or
    alignment is a finite rejection that commits nothing.
    """

    rejection_commit_policy: ClassVar[tuple[str, ...]] = ()
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    evaluation_view: EvaluationParametersFn | None = eqx.field(static=True)
    composition: ConflictFreeGradientPolicy | None
    alignment: ConflictFreeUpdatePolicy | None
    component_count: int = eqx.field(static=True)
    statistics_dtype: str = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        optimizer: optax.GradientTransformation,
        /,
        *,
        composition: ConflictFreeGradientPolicy | None,
        alignment: ConflictFreeUpdatePolicy | None,
        evaluation_parameters: EvaluationParametersFn | None,
        component_count: int,
        statistics_dtype: Any,
    ):
        if composition is None and alignment is None:
            raise ValueError("A conflict-free rule requires a composition or alignment.")
        self.optimizer = optimizer
        self.evaluation_view = evaluation_parameters
        self.composition = composition
        self.alignment = alignment
        self.component_count = component_count
        self.statistics_dtype = jnp.dtype(statistics_dtype).name
        self.rule_id = canonical_fingerprint(
            {
                "kind": "functional-conflict-free-optax",
                "composition": None if composition is None else composition.policy_id,
                "alignment": None if alignment is None else alignment.policy_id,
                "evaluation_view": evaluation_parameters is not None,
            }
        )

    def init(self, parameters: PyTree[Any], /) -> ConflictFreeOptaxState:
        if self.alignment is None:
            return ConflictFreeOptaxState(self.optimizer.init(parameters), None, None)
        # The rule state keeps the last alignment result; its structure is fixed
        # from the start by a zero result over the objective's components.
        zeros = jax.tree.map(jnp.zeros_like, parameters)
        shape = jax.eval_shape(
            lambda proposal: project_conflict_free_direction(
                proposal,
                tuple(proposal for _ in range(self.component_count)),
                policy=self.alignment,
            ),
            zeros,
        )
        return ConflictFreeOptaxState(
            self.optimizer.init(parameters),
            ConflictFreeUpdateStatistics.zeros(jnp.dtype(self.statistics_dtype)),
            jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shape),
        )

    def rule_state_finite(self, rule_state: ConflictFreeOptaxState, /) -> Array:
        # The alignment result is evidence of the last update, not an input to
        # the next one; only optimizer state and statistics gate a commit.
        finite = jnp.asarray(True)
        for leaf in jax.tree.leaves((rule_state.optimizer_state, rule_state.statistics)):
            if eqx.is_inexact_array(leaf):
                finite = finite & jnp.all(jnp.isfinite(leaf))
        return finite

    def evaluation_parameters(
        self, rule_state: ConflictFreeOptaxState, parameters: PyTree[Any], /
    ) -> PyTree[Any]:
        return resolve_evaluation_parameters(
            self.evaluation_view, rule_state.optimizer_state, parameters
        )

    def propose(
        self,
        parameters: PyTree[Any],
        gradients: PyTree[Any],
        value: Array,
        rule_state: ConflictFreeOptaxState,
        context: KernelUpdateContext,
        /,
    ) -> tuple[PyTree[Any], ConflictFreeOptaxState, ConflictFreeOptaxState, Array]:
        del value
        diagnostics = context.diagnostics[0]
        components = tuple(
            _lane_gradient(parameters, gradient)
            for gradient in diagnostics.component_gradients
        )
        successful = jnp.asarray(True)
        if self.composition is not None:
            composed = conflict_free_gradient(components, policy=self.composition)
            gradients = composed.direction
            successful = successful & composed.successful
        updates, optimizer_state = self.optimizer.update(
            gradients, rule_state.optimizer_state, parameters
        )
        statistics = rule_state.statistics
        result = None
        if self.alignment is not None:
            result = project_conflict_free_direction(
                tree_negative(updates), components, policy=self.alignment
            )
            updates = tree_where(
                result.projected, tree_negative(result.direction), updates
            )
            gradient_conflict, constructed_conflict = _alignment_conflicts(
                components, gradients, result, self.alignment
            )
            statistics = statistics.update(
                result,
                gradient_conflict=gradient_conflict,
                constructed_conflict=constructed_conflict,
            )
            successful = successful & result.successful
        candidate = optax.apply_updates(parameters, updates)
        next_state = ConflictFreeOptaxState(optimizer_state, statistics, result)
        return candidate, next_state, rule_state, successful


def _residual_builder(enforcement: Any, /) -> Any:
    """Attempt-bound residual roots of a pure ResidualPenalty objective."""

    def residual(model_state: Any, fixed: Any, prepared: Any, /) -> Any:
        held = eqx.combine(model_state, fixed)
        physical = _physical_prepared(prepared)
        terms = materialize_prepared_residual_terms(physical, require_all=True)

        def roots(parameters: Any) -> Array:
            if isinstance(prepared, PreparedFunctionalUpdate):
                if prepared.residual is None:
                    raise ValueError(
                        "Least-squares FunctionalSolver methods require residual roots."
                    )
                return prepared.residual.roots(parameters)
            pieces = tuple(
                prepared_term_residual_vector(
                    parameters,
                    held,
                    enforcement,
                    term,
                    iteration=physical.iteration,
                )
                for term in terms
            )
            if not pieces:
                raise ValueError(
                    "Least-squares FunctionalSolver methods require at least one active ResidualPenalty."
                )
            return jnp.concatenate(pieces, axis=0)

        return roots

    return residual


def _composite_problem_builder(enforcement: Any, /) -> Any:
    """Attempt-bound residual-plus-scalar problem of a composite method."""

    def problem(model_state: Any, fixed: Any, prepared: Any, /) -> Any:
        held = eqx.combine(model_state, fixed)
        physical = _physical_prepared(prepared)
        terms = materialize_prepared_residual_terms(physical)

        def residual(parameters: Any, _: Any) -> Array:
            if isinstance(prepared, PreparedFunctionalUpdate):
                if prepared.residual is None:
                    raise ValueError("GeneralizedGaussNewton requires residual roots.")
                return prepared.residual.roots(parameters)
            pieces = tuple(
                prepared_term_residual_vector(
                    parameters,
                    held,
                    enforcement,
                    term,
                    iteration=physical.iteration,
                )
                for term in terms
            )
            if not pieces:
                raise ValueError(
                    "GeneralizedGaussNewton requires at least one active ResidualPenalty."
                )
            return jnp.concatenate(pieces, axis=0)

        def scalar(parameters: Any, _: Any) -> Array:
            return evaluate_prepared_scalar_remainder(
                physical, eqx.combine(parameters, held)
            )

        return CompositeLeastSquaresProblem(
            residual, scalar, problem_id="functional-solver-composite"
        )

    return problem


def _functional_update_rule(
    route: _FunctionalOptimizerRoute,
    /,
    *,
    num_iter: int,
    enforcement: Any,
    evaluation_parameters: EvaluationParametersFn | None,
    composition: ConflictFreeGradientPolicy | None,
    alignment: ConflictFreeUpdatePolicy | None,
    statistics_dtype: Any,
    prepared_state: Any,
    component_count: int,
) -> AbstractKernelUpdateRule:
    """The one kernel update rule of a resolved FunctionalSolver optimizer route."""
    termination = OptimizationTermination(maximum_steps=num_iter)
    if route.composite is not None:
        return CompositeLeastSquaresUpdateRule(
            route.composite,
            _composite_problem_builder(enforcement),
            termination=termination,
            rule_id="functional-solver",
            prepared_state=prepared_state,
        )
    if route.least_squares is not None:
        return LeastSquaresUpdateRule(
            route.least_squares,
            _residual_builder(enforcement),
            termination=termination,
            rule_id="functional-solver",
            prepared_state=prepared_state,
        )
    if route.iterative is not None:
        return ScalarIterativeUpdateRule(
            route.iterative,
            termination=termination,
            rule_id="functional-solver",
            prepared_state=prepared_state,
        )
    if route.mirror is not None:
        return MirrorUpdateRule(route.mirror, rule_id="functional-solver")
    if route.riemannian is not None:
        if isinstance(route.riemannian, AbstractRiemannianLineSearchOptimizer):
            return RiemannianLineSearchUpdateRule(
                route.riemannian, rule_id="functional-solver"
            )
        return RiemannianUpdateRule(route.riemannian, rule_id="functional-solver")
    if route.line_search is not None:
        return OptaxUpdateRule(
            route.line_search,
            rule_id=canonical_fingerprint(
                {
                    "kind": "functional-optax-line-search",
                    "evaluation_view": evaluation_parameters is not None,
                }
            ),
            reevaluates_objective=True,
            evaluation_parameters=evaluation_parameters,
        )
    if route.standard is None:
        raise ValueError("Optimizer is not configured.")
    if composition is not None or alignment is not None:
        return _ConflictFreeOptaxRule(
            route.standard,
            composition=composition,
            alignment=alignment,
            evaluation_parameters=evaluation_parameters,
            component_count=component_count,
            statistics_dtype=statistics_dtype,
        )
    return OptaxUpdateRule(
        route.standard,
        rule_id=canonical_fingerprint(
            {
                "kind": "functional-optax",
                "evaluation_view": evaluation_parameters is not None,
            }
        ),
        evaluation_parameters=evaluation_parameters,
    )


def _prepared_native_state(
    route: _FunctionalOptimizerRoute,
    kernel: PreparedTrainingKernel,
    state: TrainingKernelState,
    payload: Any,
    loss: _FunctionalGradientLoss,
    /,
) -> Any:
    """A native method's `prepare_state` on one representative attempt payload."""
    rule = kernel.rule
    parameters = state.parameters
    if route.least_squares is not None:
        residual = rule.residual(state.model_state, kernel.fixed, payload)
        return eqx.filter_jit(route.least_squares.prepare_state)(residual, parameters)
    if route.composite is not None:
        problem = rule.problem(state.model_state, kernel.fixed, payload)
        return eqx.filter_jit(
            lambda parameters_: route.composite.prepare_state(
                problem, parameters_, args=None
            )
        )(parameters)
    if route.iterative is None:
        raise RuntimeError("Only native methods prepare a method state.")

    def value(parameters_: Any) -> Array:
        trained, held = functional_lanes(parameters_, state.model_state, kernel.fixed)
        return _objective_values(
            trained, held, payload, loss.precision, loss.surrogate_filter
        )[0]

    return eqx.filter_jit(route.iterative.prepare_state)(value, parameters)


def _objective_component_count(
    kernel: PreparedTrainingKernel,
    state: TrainingKernelState,
    payload: Any,
    loss: _FunctionalGradientLoss,
    /,
) -> int:
    """Number of gradient components of the objective on one attempt payload."""
    trained, held = functional_lanes(state.parameters, state.model_state, kernel.fixed)
    shapes = jax.eval_shape(
        lambda trained_: _objective_values(
            trained_, held, payload, loss.precision, loss.surrogate_filter
        )[2],
        trained,
    )
    return shapes.shape[0]


def _optimizer_state(rule_state: Any, /) -> Any:
    """The optimizer's own state inside a kernel rule state."""
    if isinstance(rule_state, NativeMethodState):
        return rule_state.method_state
    if isinstance(rule_state, ConflictFreeOptaxState):
        return rule_state.optimizer_state
    return rule_state


_compiled_accumulate = eqx.filter_jit(PreparedTrainingKernel.accumulate_with_diagnostics)


def _step_optimizer_metrics(
    route: _FunctionalOptimizerRoute,
    optimizer_state: Any,
    evaluation_params: Any,
    iterative_step_metrics: Any,
    update_alignment_result: ConflictFreeUpdateResult | None,
    /,
) -> dict[str, Any]:
    """Per-report optimizer scalars of one attempt."""
    mirror_step_metrics = None
    mirror_constraint_residual = None
    if route.mirror is not None:
        mirror_step_metrics = route.mirror.step_metrics(optimizer_state)
        mirror_constraint_residual = (
            route.mirror.parameter_geometry.maximum_constraint_residual(evaluation_params)
        )
    riemannian_step_metrics = None
    riemannian_constraint_residual = None
    if route.riemannian is not None:
        riemannian_step_metrics = route.riemannian.step_metrics(optimizer_state)
        riemannian_constraint_residual = (
            route.riemannian.parameter_geometry.maximum_constraint_residual(
                evaluation_params
            )
        )
    optimizer_metrics: dict[str, Any] = {}
    if iterative_step_metrics is not None:
        optimizer_metrics.update(
            {
                "optimizer/iterative/accepted_step_size": (
                    iterative_step_metrics.accepted_step_size
                ),
                "optimizer/iterative/forcing": (iterative_step_metrics.forcing),
                "optimizer/iterative/globalization_evaluations": (
                    iterative_step_metrics.globalization_evaluations
                ),
                "optimizer/iterative/linear_iterations": (
                    iterative_step_metrics.linear_iterations
                ),
                "optimizer/iterative/optimality_norm": (
                    iterative_step_metrics.optimality_norm
                ),
                "optimizer/iterative/status": (iterative_step_metrics.status),
                "optimizer/iterative/step_norm": (iterative_step_metrics.step_norm),
            }
        )
    if mirror_step_metrics is not None:
        if mirror_constraint_residual is None:
            raise RuntimeError("Mirror diagnostics require a constraint residual.")
        optimizer_metrics.update(
            {
                "optimizer/mirror/bregman_step": (mirror_step_metrics.bregman_step),
                "optimizer/mirror/constraint_residual_max": (mirror_constraint_residual),
                "optimizer/mirror/coordinate_gradient_norm": (
                    mirror_step_metrics.coordinate_gradient_norm
                ),
                "optimizer/mirror/dual_displacement_norm": (
                    mirror_step_metrics.dual_displacement_norm
                ),
                "optimizer/mirror/learning_rate": (mirror_step_metrics.learning_rate),
            }
        )
    if riemannian_step_metrics is not None:
        if riemannian_constraint_residual is None:
            raise RuntimeError("Riemannian diagnostics require a constraint residual.")
        optimizer_metrics.update(
            {
                "optimizer/riemannian/adaptive_denominator_maximum": (
                    riemannian_step_metrics.adaptive_denominator_maximum
                ),
                "optimizer/riemannian/adaptive_denominator_minimum": (
                    riemannian_step_metrics.adaptive_denominator_minimum
                ),
                "optimizer/riemannian/clipping_scale": (
                    riemannian_step_metrics.clipping_scale
                ),
                "optimizer/riemannian/conjugacy_beta": (
                    riemannian_step_metrics.conjugacy_beta
                ),
                "optimizer/riemannian/constraint_residual_max": (
                    riemannian_constraint_residual
                ),
                "optimizer/riemannian/gradient_norm": (
                    riemannian_step_metrics.gradient_norm
                ),
                "optimizer/riemannian/history_pair_count": (
                    riemannian_step_metrics.history_pair_count
                ),
                "optimizer/riemannian/learning_rate": (
                    riemannian_step_metrics.learning_rate
                ),
                "optimizer/riemannian/line_search_accepted": (
                    riemannian_step_metrics.line_search_accepted
                ),
                "optimizer/riemannian/line_search_evaluations": (
                    riemannian_step_metrics.line_search_evaluations
                ),
                "optimizer/riemannian/line_search_reduction": (
                    riemannian_step_metrics.line_search_reduction
                ),
                "optimizer/riemannian/momentum_norm": (
                    riemannian_step_metrics.momentum_norm
                ),
                "optimizer/riemannian/pair_accepted": (
                    riemannian_step_metrics.pair_accepted
                ),
                "optimizer/riemannian/restarted": (riemannian_step_metrics.restarted),
                "optimizer/riemannian/tangent_residual": (
                    riemannian_step_metrics.tangent_residual
                ),
                "optimizer/riemannian/tangent_step_norm": (
                    riemannian_step_metrics.tangent_step_norm
                ),
                "optimizer/riemannian/transport_metric_distortion": (
                    riemannian_step_metrics.transport_metric_distortion
                ),
                "optimizer/riemannian/transported_tangent_residual": (
                    riemannian_step_metrics.transported_tangent_residual
                ),
            }
        )
    if update_alignment_result is not None:
        effective_alignment = (
            update_alignment_result.active & ~update_alignment_result.stationary
        )
        minimum_raw_cosine = jnp.where(
            jnp.any(effective_alignment),
            jnp.min(
                jnp.where(
                    effective_alignment,
                    update_alignment_result.raw_cosines,
                    jnp.inf,
                )
            ),
            0.0,
        )
        minimum_aligned_cosine = jnp.where(
            jnp.any(effective_alignment),
            jnp.min(
                jnp.where(
                    effective_alignment,
                    update_alignment_result.aligned_cosines,
                    jnp.inf,
                )
            ),
            0.0,
        )
        optimizer_metrics.update(
            {
                "optimizer/update_alignment/raw_conflict": jnp.any(
                    update_alignment_result.raw_conflicts
                ),
                "optimizer/update_alignment/applied_conflict": jnp.any(
                    update_alignment_result.aligned_conflicts
                ),
                "optimizer/update_alignment/projected": (
                    update_alignment_result.projected
                ),
                "optimizer/update_alignment/minimum_raw_cosine": (minimum_raw_cosine),
                "optimizer/update_alignment/minimum_aligned_cosine": (
                    minimum_aligned_cosine
                ),
                "optimizer/update_alignment/relative_correction": (
                    update_alignment_result.relative_correction
                ),
                "optimizer/update_alignment/metric_correction_norm": (
                    update_alignment_result.metric_correction_norm
                ),
                "optimizer/update_alignment/active_constraints": (
                    update_alignment_result.active_constraint_count
                ),
                "optimizer/update_alignment/pareto_stationary": (
                    update_alignment_result.pareto_stationary
                ),
                "optimizer/update_alignment/kkt_residual": (
                    update_alignment_result.kkt_residual_norm
                ),
                "optimizer/update_alignment/status": (update_alignment_result.status),
            }
        )
    return optimizer_metrics


def _final_optimizer_diagnostics(
    route: _FunctionalOptimizerRoute, opt_state: Any, chosen: Any, /
) -> dict[str, Any]:
    """Final optimizer evidence of a finished run."""
    mirror = route.mirror
    riemannian = route.riemannian
    composite = route.composite
    least_squares = route.least_squares
    iterative = route.iterative
    mirror_diagnostics: dict[str, Any] = {}
    if mirror is not None:
        mirror_geometry = mirror.parameter_geometry
        if not bool(mirror_geometry.contains(chosen)):
            raise ValueError(
                "Returned parameters are outside their declared ParameterMirrorGeometry."
            )
        mirror_metrics = mirror.step_metrics(opt_state)
        mirror_diagnostics = {
            "optimizer/mirror/num_legendre_leaves": jnp.asarray(
                mirror_geometry.num_legendre_leaves
            ),
            "optimizer/mirror/learning_rate": mirror_metrics.learning_rate,
            "optimizer/mirror/coordinate_gradient_norm": (
                mirror_metrics.coordinate_gradient_norm
            ),
            "optimizer/mirror/dual_displacement_norm": (
                mirror_metrics.dual_displacement_norm
            ),
            "optimizer/mirror/bregman_step": mirror_metrics.bregman_step,
            "optimizer/mirror/constraint_residual_max": (
                mirror_geometry.maximum_constraint_residual(chosen)
            ),
        }
    riemannian_diagnostics: dict[str, Any] = {}
    if riemannian is not None:
        geometry = riemannian.parameter_geometry
        if not bool(geometry.contains(chosen)):
            raise ValueError(
                "Returned parameters are outside their declared ParameterGeometry."
            )
        final_metrics = riemannian.step_metrics(opt_state)
        riemannian_diagnostics = {
            "optimizer/riemannian/num_manifold_leaves": jnp.asarray(
                geometry.num_manifold_leaves
            ),
            "optimizer/riemannian/learning_rate": final_metrics.learning_rate,
            "optimizer/riemannian/gradient_norm": final_metrics.gradient_norm,
            "optimizer/riemannian/clipping_scale": final_metrics.clipping_scale,
            "optimizer/riemannian/tangent_step_norm": (final_metrics.tangent_step_norm),
            "optimizer/riemannian/momentum_norm": final_metrics.momentum_norm,
            "optimizer/riemannian/constraint_residual_max": (
                geometry.maximum_constraint_residual(chosen)
            ),
            "optimizer/riemannian/tangent_residual": (final_metrics.tangent_residual),
            "optimizer/riemannian/transported_tangent_residual": (
                final_metrics.transported_tangent_residual
            ),
            "optimizer/riemannian/transport_metric_distortion": (
                final_metrics.transport_metric_distortion
            ),
            "optimizer/riemannian/line_search_evaluations": (
                final_metrics.line_search_evaluations
            ),
            "optimizer/riemannian/line_search_accepted": (
                final_metrics.line_search_accepted
            ),
            "optimizer/riemannian/line_search_reduction": (
                final_metrics.line_search_reduction
            ),
            "optimizer/riemannian/conjugacy_beta": (final_metrics.conjugacy_beta),
            "optimizer/riemannian/history_pair_count": (final_metrics.history_pair_count),
            "optimizer/riemannian/restarted": final_metrics.restarted,
            "optimizer/riemannian/pair_accepted": final_metrics.pair_accepted,
            "optimizer/riemannian/adaptive_denominator_minimum": (
                final_metrics.adaptive_denominator_minimum
            ),
            "optimizer/riemannian/adaptive_denominator_maximum": (
                final_metrics.adaptive_denominator_maximum
            ),
        }
    iterative_diagnostics: dict[str, Any] = {}
    if composite is not None or least_squares is not None or iterative is not None:
        if composite is not None:
            final_iterative_metrics = composite.step_metrics(opt_state)
        elif least_squares is not None:
            final_iterative_metrics = least_squares.step_metrics(opt_state)
        else:
            assert iterative is not None
            final_iterative_metrics = iterative.step_metrics(opt_state)
        iterative_diagnostics = {
            "optimizer/iterative/objective": final_iterative_metrics.objective,
            "optimizer/iterative/residual_objective": (
                final_iterative_metrics.residual_objective
            ),
            "optimizer/iterative/scalar_objective": (
                final_iterative_metrics.scalar_objective
            ),
            "optimizer/iterative/optimality_norm": (
                final_iterative_metrics.optimality_norm
            ),
            "optimizer/iterative/step_norm": final_iterative_metrics.step_norm,
            "optimizer/iterative/accepted_step_size": (
                final_iterative_metrics.accepted_step_size
            ),
            "optimizer/iterative/globalization_evaluations": (
                final_iterative_metrics.globalization_evaluations
            ),
            "optimizer/iterative/accepted": final_iterative_metrics.accepted,
            "optimizer/iterative/linear_iterations": (
                final_iterative_metrics.linear_iterations
            ),
            "optimizer/iterative/linear_status": (final_iterative_metrics.linear_status),
            "optimizer/iterative/forcing": final_iterative_metrics.forcing,
            "optimizer/iterative/damping": final_iterative_metrics.damping,
            "optimizer/iterative/reduction_ratio": (
                final_iterative_metrics.reduction_ratio
            ),
            "optimizer/iterative/direction_fallback": (
                final_iterative_metrics.direction_fallback
            ),
            "optimizer/iterative/status": final_iterative_metrics.status,
        }
        if composite is not None or least_squares is not None:
            assert isinstance(opt_state, LeastSquaresState)
            iterative_diagnostics |= {
                "optimizer/iterative/iterations": opt_state.iteration,
                "optimizer/iterative/accepted_steps": opt_state.accepted_steps,
                "optimizer/iterative/rejected_steps": opt_state.rejected_steps,
                "optimizer/iterative/residual_evaluations": (
                    opt_state.residual_evaluations
                ),
                "optimizer/iterative/jvp_evaluations": opt_state.jvp_evaluations,
                "optimizer/iterative/vjp_evaluations": opt_state.vjp_evaluations,
                "optimizer/iterative/linear_solves": opt_state.linear_solves,
                "optimizer/iterative/linear_iterations_total": (
                    opt_state.linear_iterations
                ),
                "optimizer/iterative/direction_fallbacks": (
                    opt_state.direction_fallbacks
                ),
                "optimizer/iterative/scalar_evaluations": (opt_state.scalar_evaluations),
                "optimizer/iterative/scalar_gradient_evaluations": (
                    opt_state.scalar_gradient_evaluations
                ),
                "optimizer/iterative/scalar_hvp_evaluations": (
                    opt_state.scalar_hvp_evaluations
                ),
            }
        else:
            assert isinstance(opt_state, ScalarIterativeState)
            iterative_diagnostics |= {
                "optimizer/iterative/iterations": opt_state.iteration,
                "optimizer/iterative/accepted_steps": opt_state.accepted_steps,
                "optimizer/iterative/rejected_steps": opt_state.rejected_steps,
                "optimizer/iterative/objective_evaluations": (
                    opt_state.objective_evaluations
                ),
                "optimizer/iterative/gradient_evaluations": (
                    opt_state.gradient_evaluations
                ),
                "optimizer/iterative/hvp_evaluations": opt_state.hvp_evaluations,
                "optimizer/iterative/linear_solves": opt_state.linear_solves,
                "optimizer/iterative/linear_iterations_total": (
                    opt_state.linear_iterations
                ),
                "optimizer/iterative/direction_fallbacks": (
                    opt_state.direction_fallbacks
                ),
            }
    return mirror_diagnostics | riemannian_diagnostics | iterative_diagnostics


def _update_alignment_diagnostics(
    update_alignment_statistics: ConflictFreeUpdateStatistics | None, /
) -> dict[str, Any]:
    """Aggregate conflict-free alignment evidence of a finished run."""
    update_alignment_diagnostics: dict[str, Any] = {}
    if update_alignment_statistics is not None:
        update_alignment_diagnostics = {
            "optimizer/update_alignment/steps": update_alignment_statistics.steps,
            "optimizer/update_alignment/gradient_conflict_rate": (
                update_alignment_statistics.gradient_conflict_rate
            ),
            "optimizer/update_alignment/constructed_conflict_rate": (
                update_alignment_statistics.constructed_conflict_rate
            ),
            "optimizer/update_alignment/proposal_conflict_rate": (
                update_alignment_statistics.proposal_conflict_rate
            ),
            "optimizer/update_alignment/applied_conflict_rate": (
                update_alignment_statistics.applied_conflict_rate
            ),
            "optimizer/update_alignment/projection_rate": (
                update_alignment_statistics.projection_rate
            ),
            "optimizer/update_alignment/zero_proposal_steps": (
                update_alignment_statistics.zero_proposal_steps
            ),
            "optimizer/update_alignment/pareto_stationary_steps": (
                update_alignment_statistics.pareto_stationary_steps
            ),
            "optimizer/update_alignment/mean_correction_norm": (
                update_alignment_statistics.mean_correction_norm
            ),
            "optimizer/update_alignment/mean_relative_correction": (
                update_alignment_statistics.mean_relative_correction
            ),
            "optimizer/update_alignment/mean_metric_correction_norm": (
                update_alignment_statistics.mean_metric_correction_norm
            ),
            "optimizer/update_alignment/maximum_kkt_residual": (
                update_alignment_statistics.maximum_kkt_residual
            ),
        }
    return update_alignment_diagnostics


def _explicit_subspace(
    functions: Any,
    parameter_paths: tuple[str, ...],
    parameter_alias_groups: tuple[tuple[str, ...], ...],
    parameter_shapes: tuple[tuple[int, ...], ...],
    parameter_dtypes: tuple[str, ...],
    /,
) -> tuple[ParameterSubspace, Any]:
    """Verified explicit subspace and the filter of its surrogate coordinates."""
    subspace = ParameterSubspace.from_leaf_paths(
        functions,
        parameter_paths,
        alias_groups=parameter_alias_groups,
    )
    if subspace.leaf_shapes != parameter_shapes:
        raise ValueError("FunctionalSolver parameter-subspace shapes changed.")
    if subspace.leaf_dtypes != parameter_dtypes:
        raise ValueError("FunctionalSolver parameter-subspace dtypes changed.")
    validate_low_rank_subspace(functions, subspace)
    coordinate_paths = parameter_paths + tuple(
        alias for group in parameter_alias_groups for alias in group[1:]
    )
    surrogate_filter = jax.tree.map(
        lambda leaf: leaf is not None,
        ParameterSubspace.from_leaf_paths(functions, coordinate_paths).initial,
        is_leaf=lambda leaf: leaf is None,
    )
    return subspace, surrogate_filter


def solve_gradient(
    self: "FunctionalSolver",
    *,
    num_iter: int,
    optim: AbstractCompositeLeastSquaresMethod
    | AbstractLeastSquaresMethod
    | AbstractScalarIterativeMethod
    | AbstractMirrorOptimizer
    | AbstractRiemannianOptimizer
    | optax.GradientTransformation
    | optax.GradientTransformationExtraArgs
    | Any = optax.rprop(1e-3),
    evaluation_parameters: EvaluationParametersFn | None = None,
    parameter_paths: tuple[str, ...] | None = None,
    parameter_shapes: tuple[tuple[int, ...], ...] = (),
    parameter_dtypes: tuple[str, ...] = (),
    parameter_alias_groups: tuple[tuple[str, ...], ...] = (),
    seed: int = 0,
    jit: bool = True,
    keep_best: bool = True,
    log_every: int = 0,
    log_terms: bool = True,
    session: IterationSession | None = None,
    session_every: int = 1,
    tensorboard_log_dir: str | Path | None = None,
    tensorboard_every: int | None = None,
    tensorboard_flush_every: int = 10,
    profile_adaptive: bool = False,
    train_term_sample_size: int | None = None,
    gradient_accumulation: int = 1,
    precision: FunctionalPrecisionPolicy | None = None,
    training: FunctionalTrainingPlan | None = None,
    resume: bool = False,
    accepted_update_hook: Any = None,
    target_policy: DelayedTargetPolicy
    | ExponentialMovingAverageTargetPolicy
    | None = None,
) -> "FunctionalSolver":
    """Train a FunctionalSolver through the internal training kernel.

    Each loop iteration is one kernel attempt on a freshly prepared objective
    (refresh, term selection, sampling, and optional surrogate preparation stay
    on the host). The kernel owns the value and gradient, the update rule of
    the resolved optimizer route, acceptance and rollback (accepted, finite
    rejection, nonfinite), target parameters, the root key and cursors, and the
    checkpoint payload. Frontend state (pseudo-transient history, term-balance
    multipliers, diagnostic gradients) commits only with accepted updates.
    """
    accumulation_steps = int(gradient_accumulation)
    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation must be positive.")
    session_every_ = int(session_every)
    if session_every_ <= 0:
        raise ValueError("session_every must be positive.")
    if num_iter == 0:
        return self

    route = _resolve_functional_optimizer(
        optim, evaluation_parameters, parameter_paths, precision
    )

    tb_ctx = (
        _TensorBoardLogger(tensorboard_log_dir)
        if tensorboard_log_dir is not None
        else nullcontext(None)
    )

    with tb_ctx as tb_writer, _TrainingSignalGuard() as signal_guard:
        resume_state = self.training_state if resume else None
        if (
            resume_state is not None
            and resume_state.gradient_accumulation != accumulation_steps
        ):
            raise ValueError(
                "In-memory functional gradient-accumulation identity mismatch."
            )
        source_functions = (
            self.functions if resume_state is None else resume_state.current_functions
        )
        sharding_policy = None if training is None else training.sharding
        subspace, surrogate_filter = (
            (None, None)
            if parameter_paths is None
            else _explicit_subspace(
                source_functions,
                parameter_paths,
                parameter_alias_groups,
                parameter_shapes,
                parameter_dtypes,
            )
        )
        tree = functional_training_tree(
            source_functions, sharding=sharding_policy, subspace=subspace
        )
        kernel_lane = partition_parameters(tree)[0]
        route = _with_line_search_route(route, kernel_lane)
        if precision is not None and route.standard is None:
            raise ValueError(
                "Functional precision currently supports standard Optax transforms only."
            )
        if accumulation_steps > 1 and route.standard is None:
            raise ValueError(
                "gradient_accumulation > 1 is supported only by standard Optax."
            )
        if (
            accumulation_steps > 1
            and training is not None
            and (training.stateful or training.causal or training.diagnostics is not None)
        ):
            raise ValueError(
                "gradient_accumulation > 1 does not support stateful, causal, "
                "term-balancing, or diagnostic functional training policies."
            )
        trainable_dtypes = {
            leaf.dtype
            for leaf in jax.tree.leaves(kernel_lane)
            if eqx.is_inexact_array(leaf)
        }
        if precision is not None and len(trainable_dtypes) != 1:
            raise ValueError(
                "Functional precision requires one uniform trainable parameter dtype."
            )
        precision_dtype = None if precision is None else next(iter(trainable_dtypes))
        accumulation_dtypes = tuple(
            jnp.real(leaf).dtype
            for leaf in jax.tree.leaves(kernel_lane)
            if eqx.is_inexact_array(leaf)
        )
        accumulation_dtype = (
            jnp.dtype(jnp.float32)
            if not accumulation_dtypes
            else jnp.result_type(*accumulation_dtypes)
        )
        log_every_ = int(log_every)
        if log_every_ < 0:
            raise ValueError("log_every must be >= 0.")
        tb_every_ = _tensorboard_every(
            tensorboard_log_dir=tensorboard_log_dir,
            tensorboard_every=tensorboard_every,
            log_every=log_every_,
        )
        tb_flush_every_ = int(tensorboard_flush_every)
        if tb_flush_every_ <= 0:
            raise ValueError("tensorboard_flush_every must be positive.")
        log_terms_ = bool(log_terms)
        term_names = tuple(_term_label(c) for c in self.terms)
        model_loss_names = function_model_loss_labels(self.functions)
        if route.least_squares is not None and model_loss_names:
            raise ValueError(
                "Least-squares FunctionalSolver methods require a pure "
                "ResidualPenalty objective without model-level scalar losses."
            )
        gradient_composition = None if training is None else training.gradient_composition
        if gradient_composition is not None:
            if route.standard is None:
                raise ValueError(
                    "Functional gradient composition requires a standard Optax transformation."
                )
            if accumulation_steps != 1:
                raise ValueError(
                    "Functional gradient composition does not support gradient accumulation."
                )
            if train_term_sample_size not in (None, len(self.terms)):
                raise ValueError(
                    "Functional gradient composition requires every training term."
                )
            if model_loss_names:
                raise ValueError(
                    "Functional gradient composition does not yet support attached model losses."
                )
        update_alignment = None if training is None else training.update_alignment
        if update_alignment is not None:
            if route.standard is None:
                raise ValueError(
                    "Functional update alignment requires a standard Optax transformation."
                )
            if accumulation_steps != 1:
                raise ValueError(
                    "Functional update alignment does not support gradient accumulation."
                )
            if train_term_sample_size not in (None, len(self.terms)):
                raise ValueError(
                    "Functional update alignment requires every training term."
                )
            if not self.terms:
                raise ValueError(
                    "Functional update alignment requires at least one objective term."
                )
            if model_loss_names:
                raise ValueError(
                    "Functional update alignment does not yet support attached model losses."
                )
        evaluation_term_names = tuple(_term_label(c) for c in self.evaluation_terms)
        term_sample_size = _train_term_sample_size(
            train_term_sample_size,
            num_terms=len(self.terms),
        )
        loss = _FunctionalGradientLoss(
            precision,
            surrogate_filter,
            gradient_composition is not None or update_alignment is not None,
        )
        objectives = (functional_kernel_objective(loss),)

        def build_kernel(
            tree_: Any, prepared_state_: Any, component_count_: int = 0
        ) -> PreparedTrainingKernel:
            rule_ = _functional_update_rule(
                route,
                num_iter=int(num_iter),
                enforcement=self.enforcement,
                evaluation_parameters=evaluation_parameters,
                composition=gradient_composition,
                # Until the component count is known (a probe kernel that is
                # never attempted), the rule carries no alignment.
                alignment=update_alignment if component_count_ else None,
                statistics_dtype=accumulation_dtype,
                prepared_state=prepared_state_,
                component_count=component_count_,
            )
            spec_ = TrainingKernelSpec(
                rule_,
                context=_KERNEL_CONTEXT,
                rejection_budget=FUNCTIONAL_REJECTION_BUDGET,
                target_policy=target_policy,
                accumulation_dtype=accumulation_dtype,
            )
            return prepare_training_kernel(
                tree_, objectives, spec_, root_authority=FUNCTIONAL_ROOT_AUTHORITY
            )

        kernel = build_kernel(tree, None)

        def place_state(state_: TrainingKernelState) -> TrainingKernelState:
            if sharding_policy is None:
                return state_
            return replace(
                sharding_policy.place_tree(state_),
                parameters=sharding_policy.place_parameters(state_.parameters),
            )

        # Optax line searches evaluate trial points eagerly, as they always have:
        # user integrands along a line search need not be traceable.
        attempt_jit = jit and route.line_search is None

        def legacy_lanes(state_: TrainingKernelState) -> tuple[Any, Any]:
            return functional_lanes(state_.parameters, state_.model_state, kernel.fixed)

        def evaluation_view(state_: TrainingKernelState) -> Any:
            return functional_parameter_lane(
                kernel.rule.evaluation_parameters(state_.rule_state, state_.parameters)
            )

        def target_kwargs(state_: TrainingKernelState) -> dict[str, Any] | None:
            if state_.targets is None:
                return None
            return {
                "target_functions": functional_tree_functions(kernel.target_tree(state_))
            }

        def frontend_state(resume_state_: Any) -> tuple[Any, Any, Any, Any]:
            """Pseudo-transient, term-balance, and diagnostic-gradient state."""
            previous_functions_ = (
                None
                if training is None or not training.pseudo_transient
                else source_functions
                if resume_state_ is None or resume_state_.previous_functions is None
                else resume_state_.previous_functions
            )
            pseudo_inverse_steps_ = (
                ()
                if training is None
                else resume_state_.pseudo_inverse_steps
                if resume_state_ is not None
                else tuple(
                    policy.initial_inverse_step for policy in training.pseudo_transient
                )
            )
            term_multipliers_ = (
                jnp.zeros((0,), dtype=jnp.float64)
                if training is None or training.term_balance is None
                else resume_state_.term_multipliers
                if resume_state_ is not None
                else jnp.ones((len(training.term_balance.blocks),), dtype=jnp.float64)
            )
            previous_gradient_ = (
                None if resume_state_ is None else resume_state_.previous_gradient
            )
            return (
                previous_functions_,
                pseudo_inverse_steps_,
                term_multipliers_,
                previous_gradient_,
            )

        (
            previous_functions,
            pseudo_inverse_steps,
            term_multipliers,
            previous_gradient,
        ) = frontend_state(resume_state)

        def prepare_microstep(current_objective, state_, iteration):
            current_params, held = legacy_lanes(state_)
            functions_snapshot_ = _reconstruct_functions(current_params, held)
            refresh_started_ = time.perf_counter() if profile_adaptive else 0.0
            refresh_key, selection_key, evaluation_key, sampling_key = (
                functional_site_keys(
                    state_, "refresh", "term-selection", "evaluation", "sampling"
                )
            )
            refreshed = current_objective.refresh(
                functions_snapshot_,
                key=refresh_key,
                iter_=int(iteration),
            )
            refresh_elapsed = 0.0
            if profile_adaptive:
                jax.block_until_ready(refreshed)
                refresh_elapsed = time.perf_counter() - refresh_started_
            _, active_indices, term_scale = _active_train_terms(
                refreshed.terms,
                sample_size=term_sample_size,
                key=selection_key,
            )
            physical = refreshed.prepare_training(
                active_indices,
                scale=term_scale,
                evaluation_key=evaluation_key,
                sampling_key=sampling_key,
                iteration=jnp.asarray(iteration, dtype=jnp.float64),
                evaluation_kwargs=target_kwargs(state_),
            )
            if sharding_policy is not None:
                physical = sharding_policy.place_prepared(physical)
            if training is None:
                prepared_ = physical
            else:
                surrogate_params, surrogate_non_trainable = _surrogate_coordinates(
                    current_params, held, surrogate_filter
                )
                prepared_ = prepare_functional_update(
                    physical,
                    surrogate_params,
                    surrogate_non_trainable,
                    self.enforcement,
                    training=training,
                    previous_functions=previous_functions,
                    pseudo_inverse_steps=pseudo_inverse_steps,
                    term_multipliers=term_multipliers,
                    previous_gradient=previous_gradient,
                )
            return (
                refreshed,
                prepared_,
                active_indices,
                functions_snapshot_,
                refresh_elapsed,
            )

        prepared_state = None
        component_count = 0
        if update_alignment is not None:
            # The alignment rule keeps its last result in the rule state, whose
            # structure depends on the objective's component count.
            probe_state = kernel.init(tree, jr.key(seed))
            probe_payload = prepare_microstep(self.objective, probe_state, 1)[1]
            component_count = _objective_component_count(
                kernel, probe_state, probe_payload, loss
            )
            kernel = build_kernel(tree, None, component_count)
        if route.native is not None:
            # Native methods prepare their structure-dependent state once on a
            # representative attempt so the kernel's rule state keeps one
            # structure from the first attempt on.
            probe_state = kernel.init(tree, jr.key(seed))
            probe_payload = prepare_microstep(self.objective, probe_state, 1)[1]
            prepared_state = _prepared_native_state(
                route, kernel, probe_state, probe_payload, loss
            )
            kernel = build_kernel(tree, prepared_state, component_count)

        restored_objective = None
        if resume_state is not None:
            state = place_state(
                resume_functional_kernel_state(
                    kernel,
                    resume_state.kernel_state,
                    resume_state.kernel_checkpoint_id,
                )
            )
        elif resume and training is not None and training.checkpoint is not None:
            state_template = FunctionalTrainingState(
                current_functions=source_functions,
                best_functions=source_functions,
                previous_functions=(
                    source_functions if training.pseudo_transient else None
                ),
                kernel_state=kernel.init(tree, jr.key(seed)),
                kernel_checkpoint_id=kernel.checkpoint_id,
                pseudo_inverse_steps=tuple(
                    policy.initial_inverse_step for policy in training.pseudo_transient
                ),
                term_multipliers=jnp.ones(
                    (
                        0
                        if training.term_balance is None
                        else len(training.term_balance.blocks)
                    ),
                    dtype=jnp.float64,
                ),
                progress=TrainingProgress(),
                run_id=training.plan_id,
                gradient_accumulation=accumulation_steps,
            )
            restored = load_functional_training_checkpoint(
                training.checkpoint.path,
                kernel,
                self,
                state_template,
                training,
            )
            resume_state = restored.state
            restored_objective = restored.objective
            source_functions = resume_state.current_functions
            if subspace is not None:
                subspace, surrogate_filter = _explicit_subspace(
                    source_functions,
                    parameter_paths,
                    parameter_alias_groups,
                    parameter_shapes,
                    parameter_dtypes,
                )
            # The resumed run's FIXED lane is the checkpoint's; its identities
            # equal the template kernel's, which verified the restored state.
            tree = functional_training_tree(
                source_functions, sharding=sharding_policy, subspace=subspace
            )
            kernel = build_kernel(tree, prepared_state, component_count)
            (
                previous_functions,
                pseudo_inverse_steps,
                term_multipliers,
                previous_gradient,
            ) = frontend_state(resume_state)
            state = place_state(resume_state.kernel_state)
        else:
            state = kernel.init(tree, jr.key(seed))

        params, non_trainable = legacy_lanes(state)
        current_evaluation_params = evaluation_view(state)
        selection_policy = None if training is None else training.selection
        initial_progress = (
            resume_state.progress if resume_state is not None else TrainingProgress()
        )
        control = TrainingController(
            total_steps=int(num_iter),
            algorithm_id="functional-gradient-training",
            progress=initial_progress,
            session=session,
        )
        control.emit(
            TrainingIterationKind.RUN_START, metrics={"total_steps": int(num_iter)}
        )
        if resume_state is None:
            control.best_payload = current_evaluation_params
        elif subspace is None:
            control.best_payload = functional_lanes(
                *partition_parameters(
                    functional_training_tree(
                        resume_state.best_functions, sharding=sharding_policy
                    )
                )
            )[0]
        else:
            control.best_payload = ParameterSubspace.from_leaf_paths(
                resume_state.best_functions,
                parameter_paths,
                alias_groups=parameter_alias_groups,
            ).initial
        objective = self.objective if restored_objective is None else restored_objective
        if keep_best and selection_policy is not None and resume_state is None:
            initial_selection = objective.prepare_evaluation(
                key=functional_site_key(state, "initial-selection"),
                iteration=jnp.asarray(0.0),
                evaluation_kwargs=target_kwargs(state),
            )
            initial_selection_value = evaluate_prepared_objective(
                initial_selection,
                _reconstruct_functions(current_evaluation_params, non_trainable),
            ).total
            control.select(
                float(initial_selection_value),
                current_evaluation_params,
                step=0,
                mode=selection_policy.mode,
                min_delta=selection_policy.min_delta,
                patience=selection_policy.patience,
            )
        refresh_wall_time = 0.0
        optimizer_wall_time = 0.0
        first_optimizer_step_wall_time = 0.0
        steady_optimizer_step_wall_time = 0.0
        training_started = time.perf_counter()
        latest_ntk_diagnostics = None
        prepared = None
        start_update_step = (
            0 if resume_state is None else resume_state.progress.update_step
        )
        start_epoch = 0 if resume_state is None else resume_state.progress.epoch

        def make_training_state(
            state_: TrainingKernelState, selected_params: Any
        ) -> FunctionalTrainingState:
            if training is None:
                raise RuntimeError("Functional training state requires a training plan.")
            return FunctionalTrainingState(
                current_functions=functional_tree_functions(kernel.tree(state_)),
                best_functions=_reconstruct_functions(selected_params, non_trainable),
                previous_functions=previous_functions,
                kernel_state=state_,
                kernel_checkpoint_id=kernel.checkpoint_id,
                pseudo_inverse_steps=pseudo_inverse_steps,
                term_multipliers=term_multipliers,
                previous_gradient=previous_gradient,
                progress=control.progress,
                run_id=training.plan_id,
                gradient_accumulation=accumulation_steps,
                training_seconds=(
                    (0.0 if resume_state is None else resume_state.training_seconds)
                    + time.perf_counter()
                    - training_started
                ),
                resumed_from_step=start_update_step,
            )

        def publish_checkpoint(checkpoint_solver, checkpoint_state, *, final):
            if training is None or training.checkpoint is None:
                return
            if sharding_policy is not None:
                sharding_policy.synchronize(
                    f"functional-checkpoint-before-{checkpoint_state.progress.update_step}"
                )
            if sharding_policy is None or sharding_policy.is_primary_process:
                save_functional_training_checkpoint(
                    training.checkpoint.path,
                    kernel,
                    checkpoint_solver,
                    checkpoint_state,
                    training,
                    final=final,
                )
            if sharding_policy is not None:
                sharding_policy.synchronize(
                    f"functional-checkpoint-after-{checkpoint_state.progress.update_step}"
                )

        if start_epoch >= int(num_iter):
            control.emit(
                TrainingIterationKind.RUN_TERMINAL,
                metrics={"completed_steps": control.progress.update_step},
            )
            completed_state = replace(resume_state, progress=control.progress)
            resumed_result = replace_solver_state(
                self,
                functions=resume_state.best_functions,
                objective=objective,
            )
            return eqx.tree_at(
                lambda solver: solver.training_state,
                resumed_result,
                completed_state,
                is_leaf=lambda value: value is None,
            )
        for epoch in range(start_epoch, int(num_iter)):
            if control.stop_requested:
                break
            if signal_guard.stop_requested:
                _emit_training_signal_stop(
                    route.label,
                    signal_guard,
                    completed=epoch,
                    total=int(num_iter),
                )
                break
            completed = epoch
            try:
                iter_start = time.perf_counter()
                iter_ = jnp.asarray(epoch + 1, dtype=jnp.float64)
                attempt_state = state
                pre_update_params = params
                optimizer_started = time.perf_counter() if profile_adaptive else 0.0
                window_refresh_elapsed = 0.0
                if accumulation_steps == 1:
                    (
                        objective,
                        prepared,
                        active_term_indices,
                        functions_snapshot,
                        refresh_elapsed,
                    ) = prepare_microstep(objective, state, epoch + 1)
                    refresh_wall_time += refresh_elapsed
                    window_refresh_elapsed += refresh_elapsed
                    if (
                        isinstance(prepared, PreparedFunctionalUpdate)
                        and training is not None
                        and training.diagnostics is not None
                        and training.diagnostics.ntk
                        and training.diagnostics.due(epoch + 1)
                    ):
                        if prepared.residual is None:
                            raise ValueError("NTK diagnostics require residual roots.")
                        latest_ntk_diagnostics = _functional_ntk_diagnostics(
                            prepared.residual,
                            _surrogate_coordinates(
                                params, non_trainable, surrogate_filter
                            )[0],
                            training.diagnostics,
                            functional_site_key(state, "ntk"),
                        )
                    state, evidence = run_training_attempt(
                        kernel, state, prepared, jit=attempt_jit
                    )
                    control.progress = replace(
                        control.progress,
                        microstep=control.progress.microstep + 1,
                    )
                    values_arr = jnp.asarray(
                        evidence.diagnostics[0].term_values, dtype=jnp.float64
                    )
                else:
                    term_accumulators = [_ObjectiveAccumulator() for _ in term_names]
                    model_loss_accumulators = [
                        _ObjectiveAccumulator() for _ in model_loss_names
                    ]
                    for microstep in range(accumulation_steps):
                        (
                            objective,
                            prepared,
                            active_term_indices,
                            functions_snapshot,
                            refresh_elapsed,
                        ) = prepare_microstep(objective, state, epoch + 1)
                        refresh_wall_time += refresh_elapsed
                        window_refresh_elapsed += refresh_elapsed
                        if microstep + 1 < accumulation_steps:
                            state, micro_diagnostics = (
                                _compiled_accumulate(kernel, state, prepared)
                                if jit
                                else kernel.accumulate_with_diagnostics(state, prepared)
                            )
                        else:
                            state, evidence = run_training_attempt(
                                kernel, state, prepared, jit=jit
                            )
                            micro_diagnostics = evidence.diagnostics
                        micro_values = jnp.asarray(
                            micro_diagnostics[0].term_values, dtype=jnp.float64
                        )
                        active_term_count = len(prepared.terms)
                        for local_index, term_index in enumerate(active_term_indices):
                            term_accumulators[term_index] = term_accumulators[
                                term_index
                            ].add(
                                _ObjectiveContribution(
                                    micro_values[local_index],
                                    jnp.asarray(1.0, dtype=micro_values.dtype),
                                )
                            )
                        for model_index, value in enumerate(
                            micro_values[active_term_count:]
                        ):
                            model_loss_accumulators[model_index] = (
                                model_loss_accumulators[model_index].add(
                                    _ObjectiveContribution(
                                        value,
                                        jnp.asarray(1.0, dtype=micro_values.dtype),
                                    )
                                )
                            )
                        objective = objective.record_training_evaluations(
                            multiplier=1,
                            term_indices=active_term_indices,
                        )
                        control.progress = replace(
                            control.progress,
                            microstep=control.progress.microstep + 1,
                        )
                outcome = TrainingAttemptOutcome(int(evidence.outcome))
                accepted_update = outcome is TrainingAttemptOutcome.ACCEPTED
                if profile_adaptive:
                    jax.block_until_ready(state)
                    optimizer_step_wall_time = max(
                        0.0,
                        time.perf_counter() - optimizer_started - window_refresh_elapsed,
                    )
                params, non_trainable = legacy_lanes(state)
                optimizer_state = _optimizer_state(state.rule_state)
                iterative_step_metrics = (
                    None
                    if route.native is None
                    else route.native.step_metrics(optimizer_state)
                )
                update_alignment_result = (
                    state.rule_state.result
                    if accepted_update
                    and isinstance(state.rule_state, ConflictFreeOptaxState)
                    else None
                )
                if accumulation_steps == 1:
                    if accepted_update and training is not None:
                        if isinstance(prepared, PreparedFunctionalUpdate):
                            pseudo_inverse_steps = prepared.pseudo_inverse_steps
                            term_multipliers = prepared.term_multipliers
                            if prepared.diagnostic_gradient is not None:
                                previous_gradient = prepared.diagnostic_gradient
                        if training.pseudo_transient:
                            previous_functions = functions_snapshot
                    training_evaluation_multiplier = (
                        1
                        if iterative_step_metrics is None
                        else 2 + int(iterative_step_metrics.globalization_evaluations)
                    )
                    objective = objective.record_training_evaluations(
                        multiplier=training_evaluation_multiplier,
                        term_indices=active_term_indices,
                    )
                    loss_val = evidence.value
                    if accepted_update and route.evaluates_candidate:
                        # These routes judge and select the updated point, so its
                        # loss is re-read on the attempt's prepared objective.
                        loss_val, values_arr = (
                            _compiled_objective_values
                            if attempt_jit
                            else _objective_values
                        )(params, non_trainable, prepared, precision, surrogate_filter)[
                            :2
                        ]
                        values_arr = jnp.asarray(values_arr, dtype=jnp.float64)
                    active_term_count = len(prepared.terms)
                    train_term_values = _expanded_train_terms(
                        values_arr[:active_term_count],
                        active_term_indices=active_term_indices,
                        num_terms=len(term_names),
                    )
                    train_model_loss_terms = values_arr[active_term_count:]
                else:
                    loss_val = evidence.value
                    train_term_values = jnp.asarray(
                        tuple(
                            jnp.nan if accumulator.is_empty else accumulator.value
                            for accumulator in term_accumulators
                        ),
                        dtype=jnp.float64,
                    )
                    train_model_loss_terms = jnp.asarray(
                        tuple(
                            accumulator.value for accumulator in model_loss_accumulators
                        ),
                        dtype=jnp.float64,
                    )
                if profile_adaptive:
                    optimizer_wall_time += optimizer_step_wall_time
                    if epoch == 0:
                        first_optimizer_step_wall_time = optimizer_step_wall_time
                    else:
                        steady_optimizer_step_wall_time += optimizer_step_wall_time
                completed = epoch + 1
                attempt_step = epoch + 1
                control.progress = replace(control.progress, epoch=attempt_step)
                accepted_step = control.progress.update_step + int(accepted_update)
                if accepted_update:
                    control.complete_update(accepted_step)
                current_evaluation_params = evaluation_view(state)
                if accepted_update_hook is not None and accepted_update:
                    accepted_update_hook(accepted_step, current_evaluation_params)
                step = attempt_step
                evaluation_loss = None
                if accepted_update and keep_best and selection_policy is not None:
                    if selection_policy.due(accepted_step):
                        selection_prepared = objective.prepare_evaluation(
                            key=functional_site_key(attempt_state, "selection"),
                            iteration=iter_,
                        )
                        evaluation_loss = evaluate_prepared_objective(
                            selection_prepared,
                            _reconstruct_functions(
                                current_evaluation_params, non_trainable
                            ),
                        ).total
                        control.select(
                            float(evaluation_loss),
                            current_evaluation_params,
                            step=accepted_step,
                            mode=selection_policy.mode,
                            min_delta=selection_policy.min_delta,
                            patience=selection_policy.patience,
                        )
                elif accepted_update and keep_best:
                    if evaluation_parameters is None:
                        selection_parameters = (
                            params if route.evaluates_candidate else pre_update_params
                        )
                        selection_loss = loss_val
                    else:
                        evaluation_loss = (
                            _compiled_objective_values
                            if attempt_jit
                            else _objective_values
                        )(
                            current_evaluation_params,
                            non_trainable,
                            prepared,
                            precision,
                            surrogate_filter,
                        )[0]
                        selection_parameters = current_evaluation_params
                        selection_loss = evaluation_loss
                    control.select(
                        float(selection_loss),
                        selection_parameters,
                        step=accepted_step,
                    )
                log_step = (
                    _logging_enabled() and log_every_ > 0 and step % log_every_ == 0
                )
                tensorboard_step = tb_every_ is not None and (step % tb_every_ == 0)
                session_step = (
                    session is not None
                    and accepted_update
                    and accepted_step % session_every_ == 0
                )
                report_step = log_step or tensorboard_step or session_step
                iter_time_s = time.perf_counter() - iter_start
                train_data_metrics: tuple[dict[str, Any], ...] = tuple(
                    {} for _ in self.terms
                )
                eval_terms = jnp.zeros((0,), dtype=jnp.float64)
                eval_data_metrics: tuple[dict[str, Any], ...] = tuple(
                    {} for _ in self.evaluation_terms
                )
                if log_terms_ and report_step:
                    evaluation_functions = _reconstruct_functions(
                        current_evaluation_params, non_trainable
                    )
                    with _precision_context(precision):
                        active_data_metrics = prepared_data_metrics(
                            _physical_prepared(prepared), evaluation_functions
                        )
                    expanded_metrics: list[dict[str, Any]] = [
                        {} for _ in objective.training
                    ]
                    for index, metrics in zip(
                        active_term_indices,
                        active_data_metrics,
                        strict=True,
                    ):
                        expanded_metrics[index] = metrics
                    collocation_metrics = objective.collocation_data_metrics()
                    train_data_metrics = tuple(
                        data_metrics | adaptive_metrics
                        for data_metrics, adaptive_metrics in zip(
                            expanded_metrics,
                            collocation_metrics,
                            strict=True,
                        )
                    )
                    prepared_evaluation = objective.prepare_evaluation(
                        key=functional_site_key(attempt_state, "report-evaluation"),
                        iteration=iter_,
                        evaluation_kwargs=target_kwargs(state),
                    )
                    with _precision_context(precision):
                        eval_terms = evaluate_prepared_objective(
                            prepared_evaluation,
                            evaluation_functions,
                            include_model_losses=False,
                        ).term_values
                        eval_data_metrics = prepared_data_metrics(
                            prepared_evaluation, evaluation_functions
                        )

                if report_step:
                    optimizer_metrics = _step_optimizer_metrics(
                        route,
                        optimizer_state,
                        current_evaluation_params,
                        iterative_step_metrics,
                        update_alignment_result,
                    )
                    loss_f = float(loss_val)
                    best_display = _best_display_value(
                        control.progress.best_value,
                        loss_f,
                        keep_best=keep_best,
                    )
                    scalars = _training_scalars(
                        loss=loss_f,
                        best_loss=best_display,
                        evaluation_loss=evaluation_loss,
                        iter_time_s=iter_time_s,
                        train_term_names=term_names,
                        train_terms=train_term_values,
                        train_data_metrics=train_data_metrics,
                        train_model_loss_names=model_loss_names,
                        train_model_loss_terms=train_model_loss_terms,
                        evaluation_term_names=evaluation_term_names,
                        eval_terms=eval_terms,
                        eval_data_metrics=eval_data_metrics,
                        log_terms=log_terms_,
                        optimizer_metrics=optimizer_metrics,
                    )
                    if session_step:
                        control.deliver(
                            TrainingIterationKind.UPDATE,
                            metrics=scalars,
                        )
                    if log_step:
                        _emit_training_scalars(
                            scalars,
                            backend=route.label,
                            step=step,
                            total_steps=int(num_iter),
                        )
                    if tensorboard_step and tb_writer is not None:
                        _write_tensorboard_scalars(tb_writer, scalars, step=step)
                        if step % tb_flush_every_ == 0:
                            tb_writer.flush()
                if (
                    accepted_update
                    and training is not None
                    and training.checkpoint is not None
                    and training.checkpoint.due(accepted_step)
                ):
                    checkpoint_state = make_training_state(
                        state,
                        control.selected(current_evaluation_params)
                        if keep_best
                        else current_evaluation_params,
                    )
                    checkpoint_solver = eqx.tree_at(
                        lambda solver: solver.training_state,
                        replace_solver_state(
                            self,
                            functions=checkpoint_state.best_functions,
                            objective=objective,
                        ),
                        checkpoint_state,
                        is_leaf=lambda value: value is None,
                    )
                    publish_checkpoint(checkpoint_solver, checkpoint_state, final=False)
                if control.stop_requested:
                    break
                if signal_guard.stop_requested:
                    _emit_training_signal_stop(
                        route.label,
                        signal_guard,
                        completed=step,
                        total=int(num_iter),
                    )
                    break
                if iterative_step_metrics is not None and int(
                    iterative_step_metrics.status
                ) != int(OptimizationStatus.ITERATING):
                    break
            except (KeyboardInterrupt, InterruptedError) as exc:
                signal_guard.request_stop_from_exception(exc)
                _emit_training_signal_stop(
                    route.label,
                    signal_guard,
                    completed=completed,
                    total=int(num_iter),
                )
                break

        params, non_trainable = legacy_lanes(state)
        current_evaluation_params = evaluation_view(state)
        chosen = (
            control.selected(current_evaluation_params)
            if keep_best
            else current_evaluation_params
        )
        optimizer_diagnostics = _final_optimizer_diagnostics(
            route, _optimizer_state(state.rule_state), chosen
        )
        functions = _reconstruct_functions(chosen, non_trainable)
        settle_started = time.perf_counter() if profile_adaptive else 0.0
        with _precision_context(precision):
            objective = objective.settle(
                functions,
                key=functional_site_key(state, "settle"),
                iter_=completed + 1,
            )
        if profile_adaptive:
            jax.block_until_ready(objective)
            refresh_wall_time += time.perf_counter() - settle_started
        result = replace_solver_state(
            self,
            functions=functions,
            objective=objective,
        )
        precision_evidence = (
            None if precision is None else precision.evidence(precision_dtype)
        )
        result = result._with_precision_evidence(
            precision,
            precision_evidence,
        )
        control.emit(
            TrainingIterationKind.RUN_TERMINAL,
            metrics={"completed_steps": completed},
        )
        if training is not None:
            training_state = make_training_state(state, chosen)
            result = eqx.tree_at(
                lambda solver: solver.training_state,
                result,
                training_state,
                is_leaf=lambda value: value is None,
            )
            if training.checkpoint is not None and training.checkpoint.save_final:
                publish_checkpoint(result, training_state, final=True)
        objective_plane_diagnostics: dict[str, Any] = {}
        if isinstance(prepared, PreparedFunctionalUpdate):
            chosen_surrogate, chosen_surrogate_non_trainable = _surrogate_coordinates(
                chosen, non_trainable, surrogate_filter
            )
            objective_plane_diagnostics = {
                "objective/physical": prepared.physical_values(functions).total,
                "objective/surrogate": prepared.surrogate_loss(
                    chosen_surrogate,
                    chosen_surrogate_non_trainable,
                ),
                "gradient_alignment/intra": prepared.intra_gradient_alignment,
                "gradient_alignment/inter": prepared.inter_gradient_alignment,
            }
        alignment_statistics = (
            state.rule_state.statistics
            if isinstance(state.rule_state, ConflictFreeOptaxState)
            else None
        )
        diagnostics = frozendict(
            {
                "profile_enabled": jnp.asarray(profile_adaptive),
                "refresh_wall_time_seconds": jnp.asarray(refresh_wall_time),
                "optimizer_wall_time_seconds": jnp.asarray(optimizer_wall_time),
                "optimizer_first_step_wall_time_seconds": jnp.asarray(
                    first_optimizer_step_wall_time
                ),
                "optimizer_steady_step_wall_time_seconds": jnp.asarray(
                    steady_optimizer_step_wall_time / max(completed - 1, 1)
                ),
                "optimizer/rejected_attempts_finite": state.finite_rejections,
                "optimizer/rejected_attempts_nonfinite": state.nonfinite_rejections,
            }
            | objective_plane_diagnostics
            | _functional_ntk_diagnostic_values(latest_ntk_diagnostics)
            | optimizer_diagnostics
            | _update_alignment_diagnostics(alignment_statistics)
        )
        return eqx.tree_at(lambda s: s.training_diagnostics, result, diagnostics)


__all__ = ["solve_gradient"]
