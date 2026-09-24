#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phydrax-native KFAC over frozen residual terms, trained by the kernel.

Every attempt freezes one prepared objective (the payload). The kernel objective
evaluates its ordered total and emits the attempt's block curvature observations
as diagnostics; `KFACUpdateRule` folds them into the Kronecker factors, solves
the damped block system for the kernel gradient, and searches the step on the
same frozen realization through `KernelUpdateContext.objective_value`.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree

from .._fingerprint import canonical_fingerprint
from .._frozendict import frozendict
from .._iteration import IterationSession
from .._strict import StrictModule
from .._training import (
    emit_training_signal_stop as _emit_training_signal_stop,
    tensorboard_every as _tensorboard_every,
    TensorBoardLogger,
    TrainingController,
    TrainingIterationKind,
    TrainingProgress,
    TrainingSignalGuard as _TrainingSignalGuard,
)
from .._training_kernel import (
    AbstractKernelUpdateRule,
    KernelUpdateContext,
    prepare_training_kernel,
    PreparedTrainingKernel,
    run_training_attempt,
    TrainingAttemptOutcome,
    TrainingKernelSpec,
    TrainingKernelState,
)
from .._tree_math import tree_allfinite, tree_where
from ..logging import is_enabled as _logging_enabled
from ..optim._iterative._globalization import (
    armijo_backtracking,
    ArmijoLineSearch,
)
from ..optim._kfac._blocks import (
    solve_block_direction,
    update_block_state_from_observations,
)
from ..optim._kfac._config import KFAC
from ..optim._kfac._types import (
    BlockCurvatureObservation,
    KFACMetrics,
    KFACState,
    ParameterLayout,
)
from ._functional_checkpoint import (
    load_functional_training_checkpoint,
    save_functional_training_checkpoint,
)
from ._functional_kernel import (
    functional_kernel_objective,
    functional_lanes,
    FUNCTIONAL_REJECTION_BUDGET,
    FUNCTIONAL_ROOT_AUTHORITY,
    functional_site_key,
    functional_training_tree,
    resume_functional_kernel_state,
)
from ._functional_objective import (
    evaluate_prepared_objective,
    prepared_data_metrics,
)
from ._functional_reporting import (
    emit_training_scalars as _emit_training_scalars,
    term_label as _term_label,
    training_scalars as _training_scalars,
    write_tensorboard_scalars as _write_tensorboard_scalars,
)
from ._functional_residual import materialize_prepared_residual_terms
from ._functional_run import (
    expand_train_terms as _expanded_train_terms,
    partition_functional_parameters,
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
from ._functional_training import FunctionalTrainingPlan, FunctionalTrainingState
from ._kfac_layout import build_kfac_plan, KFACPlan
from ._kfac_problem import (
    term_block_curvature_observations,
    validate_derivative_coverage,
)
from ._model_losses import function_model_loss_labels


def _armijo_search(
    flat_parameters,
    direction,
    gradient,
    initial_loss,
    loss_function,
    /,
    *,
    learning_rate: float,
    shrink: float,
    c1: float,
    max_steps: int,
) -> tuple[Any, Any, Any, Any, Any]:
    """Armijo backtracking along `-direction`: `(parameters, value, rate, steps, ok)`.

    A zero direction (zero gradient) is an accepted zero step. A nonzero
    direction that is not a finite descent direction, or a search that saw no
    finite candidate, is not accepted. A search that saw finite candidates but no
    sufficient decrease is an accepted zero step.
    """
    directional_derivative = jnp.vdot(gradient, direction).real
    zero_direction = jnp.linalg.norm(direction) == 0.0
    descent = (
        jnp.isfinite(initial_loss)
        & jnp.isfinite(directional_derivative)
        & (directional_derivative > 0.0)
    )
    minimum_rate = float(learning_rate) * float(shrink) ** (int(max_steps) - 1)
    if minimum_rate == 0.0:
        minimum_rate = float(learning_rate)
    policy = ArmijoLineSearch(
        initial_rate=learning_rate,
        contraction=shrink,
        sufficient_decrease=c1,
        maximum_steps=max_steps,
        minimum_rate=minimum_rate,
    )
    value_dtype = jnp.asarray(initial_loss).dtype
    rate_dtype = jnp.result_type(initial_loss, directional_derivative, jnp.float64)

    def no_search(_):
        return (
            flat_parameters,
            jnp.asarray(initial_loss, dtype=value_dtype),
            jnp.zeros((), dtype=rate_dtype),
            jnp.asarray(0, dtype=jnp.int32),
            zero_direction,
        )

    def search(_):
        result = armijo_backtracking(
            loss_function,
            flat_parameters,
            initial_loss,
            -direction,
            -directional_derivative,
            step=lambda base, tangent, rate: base + (rate * tangent).astype(base.dtype),
            contains=lambda candidate: jnp.all(jnp.isfinite(candidate)),
            policy=policy,
        )
        return (
            result.parameters,
            result.value.astype(value_dtype),
            result.rate.astype(rate_dtype),
            result.evaluations,
            result.finite_candidate_seen,
        )

    return jax.lax.cond(descent & ~zero_direction, search, no_search, None)


def _fixed_step(flat_parameters, direction, loss_function, /, *, learning_rate: float):
    """One fixed step `x - learning_rate * direction`, accepted iff its loss is finite."""
    candidate = flat_parameters - (float(learning_rate) * direction).astype(
        flat_parameters.dtype
    )
    candidate_loss = loss_function(candidate)
    return (
        candidate,
        candidate_loss,
        jnp.asarray(float(learning_rate), dtype=candidate_loss.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.isfinite(candidate_loss),
    )


def _quadratic_norm_and_clip(direction, gradient, /, *, maximum: float | None):
    quadratic_norm = jnp.sqrt(jnp.maximum(jnp.vdot(gradient, direction).real, 0.0))
    if maximum is not None:
        ratio = float(maximum) / jnp.maximum(quadratic_norm, 1e-30)
        scale = jnp.minimum(1.0, ratio)
        direction = scale * direction
        quadratic_norm = scale * quadratic_norm
    return direction, quadratic_norm


def _regularized_psd_condition(matrix, /, *, damping: float):
    eigenvalues = jnp.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    regularizer = jnp.sqrt(float(damping))
    return (jnp.max(eigenvalues) + regularizer) / (
        jnp.maximum(jnp.min(eigenvalues), 0.0) + regularizer
    )


def _factor_condition_estimate(curvature, /, *, damping: float):
    maximum = jnp.asarray(1.0)
    for block_terms in curvature.affine:
        for factor in block_terms:
            condition = _regularized_psd_condition(
                factor.activation,
                damping=damping,
            ) * _regularized_psd_condition(
                factor.sensitivity,
                damping=damping,
            )
            maximum = jnp.maximum(maximum, condition)
    for factor in curvature.uncovered:
        if factor.value.ndim == 2:
            condition = _regularized_psd_condition(
                factor.value,
                damping=damping**2,
            )
        else:
            diagonal = factor.value + float(damping)
            condition = jnp.max(diagonal) / jnp.min(diagonal)
        maximum = jnp.maximum(maximum, condition)
    return maximum


# Kernel objective and update rule ----------------------------------------------------


@final
class KFACCurvatureObservations(StrictModule):
    """KFAC objective diagnostics: block curvature observations of the active terms.

    `term_indices` are the original objective indices of the observed terms; both
    are empty on an attempt that does not refresh the factors.
    """

    observations: tuple[BlockCurvatureObservation, ...]
    term_indices: tuple[int, ...] = eqx.field(static=True)


@final
class _KFACPayload(StrictModule):
    """One attempt: the frozen objective and whether it refreshes the curvature.

    `objective` is the prepared physical objective, or (under a training plan)
    the same-update surrogate carrying only its physical objective and residual.
    """

    objective: Any
    refresh: bool = eqx.field(static=True)


@final
class _KFACLoss(StrictModule):
    """Frozen functional loss of one KFAC attempt plus its curvature observations.

    The observations are formed from the parameters the kernel passes (under its
    admission mask) with the tangent stopped: they are statistics of the
    committed parameters, not part of the differentiated objective.
    """

    layout: ParameterLayout = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)

    def __call__(
        self, parameters: Any, held: Any, payload: _KFACPayload, keys: Any
    ) -> tuple[Any, KFACCurvatureObservations]:
        del keys
        objective = payload.objective
        if isinstance(objective, PreparedFunctionalUpdate):
            total = objective.surrogate_loss(parameters, held)
            residual = objective.residual
            terms = residual.terms
            enforcement = residual.enforcement
        else:
            total = evaluate_prepared_objective(
                objective, eqx.combine(parameters, held)
            ).total
            residual = None
            terms = materialize_prepared_residual_terms(objective, require_all=True)
            enforcement = objective.enforcement
        if not payload.refresh:
            return total, KFACCurvatureObservations((), ())
        _, observations = term_block_curvature_observations(
            jax.lax.stop_gradient(parameters),
            held,
            enforcement,
            terms,
            self.layout,
            approximation=self.approximation,
            chunk_size=self.chunk_size,
            iter_=objective.iteration,
            functional_residual=residual,
        )
        return total, KFACCurvatureObservations(
            observations, tuple(term.index for term in terms)
        )


def _like(value: Any, reference: Any, /) -> Any:
    return jnp.asarray(value).astype(jnp.asarray(reference).dtype)


@final
class KFACUpdateRule(AbstractKernelUpdateRule):
    """Phydrax-native type-II GGN KFAC update of one frozen functional objective.

    Each attempt folds the objective's curvature observations into the per-term
    Kronecker factors, solves the damped block system for the kernel gradient,
    clips the quadratic update norm, and takes an Armijo search (or a fixed step)
    on the attempt's frozen objective via `context.objective_value`.

    A zero direction (zero gradient) and an Armijo search that saw finite
    candidates without sufficient decrease are accepted zero steps. A nonfinite
    curvature, gradient, or direction, a nonzero non-descent direction, a search
    whose every candidate was nonfinite, and a nonfinite fixed-step loss are
    rejections. On a finite rejection `rejection_commit_policy` keeps the updated
    curvature and its `factor_updates` counter: they are statistics of the
    attempt's realization at the committed parameters, so a retry keeps
    accumulating curvature. The accepted-step counter and the step metrics
    (derived from the rejected proposal) are not committed. A nonfinite curvature
    makes the rejection state nonfinite, so the kernel rolls it back.
    """

    rejection_commit_policy: ClassVar[tuple[str, ...]] = (
        "curvature",
        "factor_updates",
    )
    plan: KFACPlan = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(self, plan: KFACPlan, /):
        if not isinstance(plan, KFACPlan):
            raise TypeError("plan must be a KFACPlan.")
        self.plan = plan
        self.rule_id = canonical_fingerprint(
            {
                "kind": "functional-kfac",
                "config": asdict(plan.config),
                "num_terms": plan.num_terms,
            }
        )

    def init(self, parameters: Any, /) -> KFACState:
        return self.plan.initialize(parameters)

    def propose(
        self,
        parameters: Any,
        gradients: Any,
        value: Any,
        rule_state: KFACState,
        context: KernelUpdateContext,
        /,
    ) -> tuple[Any, KFACState, KFACState, Any]:
        config = self.plan.config
        (observed,) = context.diagnostics
        curvature = rule_state.curvature
        factor_updates = rule_state.factor_updates
        if observed.term_indices:
            curvature = jax.tree.map(
                _like,
                update_block_state_from_observations(
                    curvature,
                    observed.observations,
                    factor_decay=config.factor_decay,
                    term_indices=observed.term_indices,
                ),
                rule_state.curvature,
            )
            factor_updates = factor_updates + 1
        flat_parameters, unravel = ravel_pytree(parameters)
        flat_gradient = ravel_pytree(gradients)[0].astype(flat_parameters.dtype)
        curvature_finite = tree_allfinite(curvature)
        gradient_finite = jnp.isfinite(value) & jnp.all(jnp.isfinite(flat_gradient))
        # The block solve only ever sees finite inputs (the committed curvature is
        # finite): nonfinite curvature or gradients reject the attempt instead of
        # tripping the solver's finiteness checks.
        direction, cg_iterations, cg_relative_residual = solve_block_direction(
            tree_where(curvature_finite, curvature, rule_state.curvature),
            self.plan.layout,
            jnp.where(gradient_finite, flat_gradient, 0.0),
            damping=config.damping,
            cg_max_steps=config.cg_max_steps,
            cg_relative_tolerance=config.cg_relative_tolerance,
        )
        usable = curvature_finite & gradient_finite & jnp.all(jnp.isfinite(direction))
        gradient = jnp.where(usable, flat_gradient, 0.0)
        direction, quadratic_norm = _quadratic_norm_and_clip(
            jnp.where(usable, direction, 0.0),
            gradient,
            maximum=config.max_update_norm,
        )

        def objective_value(flat_candidate):
            return context.objective_value(
                unravel(flat_candidate.astype(flat_parameters.dtype))
            )

        if config.line_search:
            candidate, candidate_value, step_size, steps, step_accepted = _armijo_search(
                flat_parameters,
                direction,
                gradient,
                value,
                objective_value,
                learning_rate=config.learning_rate,
                shrink=config.line_search_shrink,
                c1=config.line_search_c1,
                max_steps=config.line_search_max_steps,
            )
        else:
            candidate, candidate_value, step_size, steps, step_accepted = _fixed_step(
                flat_parameters,
                direction,
                objective_value,
                learning_rate=config.learning_rate,
            )
        metrics = rule_state.metrics
        candidate_state = KFACState(
            step=rule_state.step + 1,
            curvature=curvature,
            factor_updates=factor_updates,
            metrics=KFACMetrics(
                cg_iterations_max=_like(cg_iterations, metrics.cg_iterations_max),
                cg_relative_residual_max=_like(
                    cg_relative_residual, metrics.cg_relative_residual_max
                ),
                quadratic_update_norm=_like(
                    quadratic_norm, metrics.quadratic_update_norm
                ),
                accepted_step_size=_like(step_size, metrics.accepted_step_size),
                line_search_steps=_like(steps, metrics.line_search_steps),
                candidate_value=_like(candidate_value, metrics.candidate_value),
            ),
        )
        rejection_state = KFACState(
            step=rule_state.step,
            curvature=curvature,
            factor_updates=factor_updates,
            metrics=metrics,
        )
        return (
            unravel(candidate.astype(flat_parameters.dtype)),
            candidate_state,
            rejection_state,
            usable & step_accepted,
        )


def _prepare_kfac_kernel(
    tree: Any, plan: KFACPlan, optim: KFAC, /
) -> PreparedTrainingKernel:
    return prepare_training_kernel(
        tree,
        (
            functional_kernel_objective(
                _KFACLoss(
                    layout=plan.layout,
                    approximation=optim.approximation,
                    chunk_size=int(optim.factor_chunk_size),
                )
            ),
        ),
        TrainingKernelSpec(
            KFACUpdateRule(plan),
            context="FunctionalSolver.solve(KFAC)",
            rejection_budget=FUNCTIONAL_REJECTION_BUDGET,
        ),
        root_authority=FUNCTIONAL_ROOT_AUTHORITY,
    )


# Frontend ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _KFACSetup:
    resume_state: FunctionalTrainingState | None
    kernel: PreparedTrainingKernel
    kernel_state: TrainingKernelState
    plan: KFACPlan
    objective: Any


def _validate_kfac_configuration(
    self,
    evaluation_parameters,
    log_every: int,
    tensorboard_flush_every: int,
    session_every: int,
    /,
) -> int:
    if evaluation_parameters is not None:
        raise ValueError("evaluation_parameters is not supported by KFAC.")
    if int(log_every) < 0:
        raise ValueError("log_every must be >= 0.")
    if int(tensorboard_flush_every) <= 0:
        raise ValueError("tensorboard_flush_every must be positive.")
    session_every_ = int(session_every)
    if session_every_ <= 0:
        raise ValueError("session_every must be positive.")
    coverage_functions = (
        self.functions
        if self.enforcement is None
        else self.enforcement.apply(self.functions)
    )
    validate_derivative_coverage(self.terms, coverage_functions)
    model_loss_labels = function_model_loss_labels(self.functions)
    if model_loss_labels:
        raise ValueError(
            "KFAC does not support attached model losses because they do not provide "
            f"residual roots; found {', '.join(model_loss_labels)}."
        )
    return session_every_


def _prepare_kfac_setup(
    self,
    optim: KFAC,
    seed: int,
    training: FunctionalTrainingPlan | None,
    resume: bool,
    /,
) -> _KFACSetup:
    resume_state = self.training_state if resume else None
    source_functions = (
        self.functions if resume_state is None else resume_state.current_functions
    )
    sharding_policy = None if training is None else training.sharding
    params, _ = partition_functional_parameters(
        source_functions, sharding=sharding_policy
    )
    plan = build_kfac_plan(
        optim,
        source_functions,
        params,
        num_terms=len(self.terms),
    )
    tree = functional_training_tree(source_functions, sharding=sharding_policy)
    kernel = _prepare_kfac_kernel(tree, plan, optim)
    objective = self.objective
    if resume_state is not None:
        kernel_state = resume_functional_kernel_state(
            kernel, resume_state.kernel_state, resume_state.kernel_checkpoint_id
        )
        return _KFACSetup(resume_state, kernel, kernel_state, plan, objective)
    kernel_state = kernel.init(tree, jr.key(int(seed)))
    if resume and training is not None and training.checkpoint is not None:
        state_template = FunctionalTrainingState(
            current_functions=source_functions,
            best_functions=source_functions,
            previous_functions=(source_functions if training.pseudo_transient else None),
            kernel_state=kernel_state,
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
        )
        restored = load_functional_training_checkpoint(
            training.checkpoint.path,
            kernel,
            self,
            state_template,
            training,
        )
        resume_state = restored.state
        kernel_state = resume_state.kernel_state
        if sharding_policy is not None:
            kernel_state = eqx.tree_at(
                lambda state: state.rule_state,
                kernel_state,
                sharding_policy.place_tree(kernel_state.rule_state),
            )
        objective = restored.objective
    return _KFACSetup(resume_state, kernel, kernel_state, plan, objective)


def solve_kfac(
    self,
    *,
    num_iter: int,
    optim: KFAC,
    evaluation_parameters,
    seed: int,
    jit: bool,
    keep_best: bool,
    log_every: int,
    log_terms: bool,
    session: IterationSession | None = None,
    session_every: int = 1,
    tensorboard_log_dir: str | Path | None,
    tensorboard_every: int | None,
    tensorboard_flush_every: int,
    profile_adaptive: bool,
    train_term_sample_size: int | None,
    training: FunctionalTrainingPlan | None = None,
    resume: bool = False,
):
    """Run Phydrax-native KFAC over frozen residual terms.

    `num_iter` bounds kernel attempts (`TrainingProgress.epoch`); accepted
    updates advance `TrainingProgress.update_step`. Reporting, selection, and
    periodic checkpoints happen at accepted updates.
    """

    if int(num_iter) < 0:
        raise ValueError("num_iter must be non-negative.")
    if int(num_iter) == 0:
        return self
    session_every_ = _validate_kfac_configuration(
        self,
        evaluation_parameters,
        log_every,
        tensorboard_flush_every,
        session_every,
    )
    setup = _prepare_kfac_setup(self, optim, seed, training, resume)
    resume_state = setup.resume_state
    kernel = setup.kernel
    kernel_state = setup.kernel_state
    plan = setup.plan
    objective = setup.objective
    sharding_policy = None if training is None else training.sharding
    _, held = functional_lanes(
        kernel_state.parameters, kernel_state.model_state, kernel.fixed
    )
    term_sample_size = _train_term_sample_size(
        train_term_sample_size,
        num_terms=len(self.terms),
    )
    term_names = tuple(_term_label(term) for term in self.terms)
    evaluation_term_names = tuple(_term_label(term) for term in self.evaluation_terms)
    selection_policy = None if training is None else training.selection
    initial_progress = (
        TrainingProgress() if resume_state is None else resume_state.progress
    )
    control = TrainingController(
        total_steps=int(num_iter),
        algorithm_id="functional-kfac-training",
        progress=initial_progress,
        session=session,
    )
    control.best_payload = (
        kernel_state.parameters
        if resume_state is None
        else partition_functional_parameters(resume_state.best_functions)[0]
    )
    best_loss = (
        float("inf")
        if control.progress.best_value is None
        else float(control.progress.best_value)
    )
    if keep_best and selection_policy is not None and resume_state is None:
        initial_selection = objective.prepare_evaluation(
            key=functional_site_key(kernel_state, "initial-selection"),
            iteration=jnp.asarray(0.0),
        )
        initial_value = evaluate_prepared_objective(
            initial_selection, eqx.combine(kernel_state.parameters, held)
        ).total
        control.select(
            float(initial_value),
            kernel_state.parameters,
            step=0,
            mode=selection_policy.mode,
            min_delta=selection_policy.min_delta,
            patience=selection_policy.patience,
        )
        best_loss = float(initial_value)
    control.emit(
        TrainingIterationKind.RUN_START,
        metrics={"total_steps": int(num_iter)},
    )
    start_epoch = control.progress.epoch
    start_step = control.progress.update_step
    if start_epoch >= int(num_iter):
        control.emit(
            TrainingIterationKind.RUN_TERMINAL,
            metrics={"completed_steps": control.progress.update_step},
        )
        completed_state = replace(
            resume_state, kernel_state=kernel_state, progress=control.progress
        )
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
    previous_functions = (
        None
        if training is None or not training.pseudo_transient
        else eqx.combine(kernel_state.parameters, held)
        if resume_state is None or resume_state.previous_functions is None
        else resume_state.previous_functions
    )
    pseudo_inverse_steps = (
        ()
        if training is None
        else resume_state.pseudo_inverse_steps
        if resume_state is not None
        else tuple(policy.initial_inverse_step for policy in training.pseudo_transient)
    )
    term_multipliers = (
        jnp.zeros((0,), dtype=jnp.float64)
        if training is None or training.term_balance is None
        else resume_state.term_multipliers
        if resume_state is not None
        else jnp.ones((len(training.term_balance.blocks),), dtype=jnp.float64)
    )
    previous_gradient = None if resume_state is None else resume_state.previous_gradient
    refresh_wall_time = 0.0
    optimizer_wall_time = 0.0
    first_optimizer_step_wall_time = 0.0
    steady_optimizer_step_wall_time = 0.0
    attempts = 0
    training_started = time.perf_counter()
    latest_ntk_diagnostics = None
    update = None

    def make_training_state(selected_params):
        if training is None:
            raise RuntimeError("Functional training state requires a training plan.")
        return FunctionalTrainingState(
            current_functions=eqx.combine(kernel_state.parameters, held),
            best_functions=eqx.combine(selected_params, held),
            previous_functions=previous_functions,
            kernel_state=kernel_state,
            kernel_checkpoint_id=kernel.checkpoint_id,
            pseudo_inverse_steps=pseudo_inverse_steps,
            term_multipliers=term_multipliers,
            previous_gradient=previous_gradient,
            progress=control.progress,
            run_id=training.plan_id,
            training_seconds=(
                (0.0 if resume_state is None else resume_state.training_seconds)
                + time.perf_counter()
                - training_started
            ),
            resumed_from_step=start_step,
        )

    def publish_checkpoint(checkpoint_solver, checkpoint_state, *, final=False):
        if training is None or training.checkpoint is None:
            return
        if sharding_policy is not None:
            sharding_policy.synchronize(
                f"functional-kfac-checkpoint-before-{checkpoint_state.progress.update_step}"
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
                f"functional-kfac-checkpoint-after-{checkpoint_state.progress.update_step}"
            )

    tensorboard_context = (
        TensorBoardLogger(tensorboard_log_dir)
        if tensorboard_log_dir is not None
        else nullcontext(None)
    )
    tensorboard_period = _tensorboard_every(
        tensorboard_log_dir=tensorboard_log_dir,
        tensorboard_every=tensorboard_every,
        log_every=int(log_every),
    )

    with (
        tensorboard_context as tensorboard_writer,
        _TrainingSignalGuard() as signal_guard,
    ):
        for epoch in range(start_epoch, int(num_iter)):
            if control.stop_requested:
                break
            if signal_guard.stop_requested:
                _emit_training_signal_stop(
                    "kfac",
                    signal_guard,
                    completed=control.progress.update_step,
                    total=int(num_iter),
                )
                break
            iteration_started = time.perf_counter()
            iteration = epoch + 1
            term_iteration = jnp.asarray(iteration, dtype=jnp.float64)
            # Every host site of this attempt is addressed by its attempt cursor.
            attempt_state = kernel_state
            params = attempt_state.parameters
            functions_snapshot = eqx.combine(params, held)
            refresh_started = time.perf_counter()
            objective = objective.refresh(
                functions_snapshot,
                key=functional_site_key(attempt_state, "refresh"),
                iter_=term_iteration,
            )
            if profile_adaptive:
                jax.block_until_ready(objective)
                refresh_wall_time += time.perf_counter() - refresh_started

            if epoch == 0:
                active_indices = tuple(range(len(self.terms)))
                term_scale = jnp.asarray(1.0)
            else:
                _, active_indices, term_scale = _active_train_terms(
                    self.terms,
                    sample_size=term_sample_size,
                    key=functional_site_key(attempt_state, "term-selection"),
                )
            prepared = objective.prepare_training(
                active_indices,
                scale=term_scale,
                evaluation_key=functional_site_key(attempt_state, "evaluation"),
                sampling_key=functional_site_key(attempt_state, "sampling"),
                iteration=term_iteration,
            )
            if sharding_policy is not None:
                prepared = sharding_policy.place_prepared(prepared)
            update = (
                None
                if training is None
                else prepare_functional_update(
                    prepared,
                    params,
                    held,
                    self.enforcement,
                    training=training,
                    previous_functions=previous_functions,
                    pseudo_inverse_steps=pseudo_inverse_steps,
                    term_multipliers=term_multipliers,
                    previous_gradient=previous_gradient,
                )
            )
            if (
                update is not None
                and training.diagnostics is not None
                and training.diagnostics.ntk
                and training.diagnostics.due(iteration)
            ):
                if update.residual is None:
                    raise ValueError("NTK diagnostics require residual roots.")
                latest_ntk_diagnostics = _functional_ntk_diagnostics(
                    update.residual,
                    params,
                    training.diagnostics,
                    functional_site_key(attempt_state, "ntk"),
                )
            payload = _KFACPayload(
                (
                    prepared
                    if update is None
                    else PreparedFunctionalUpdate(update.physical, update.residual)
                ),
                refresh=epoch % optim.factor_update_period == 0,
            )
            optimizer_started = time.perf_counter()
            # KFAC always compiles its attempt; `jit` is recorded, not honored
            # (docs/api/optim.md), so eager and jit requests replay bitwise.
            kernel_state, evidence = run_training_attempt(kernel, kernel_state, payload)
            outcome, attempt_value, candidate_value = jax.device_get(
                (
                    evidence.outcome,
                    evidence.value,
                    kernel_state.rule_state.metrics.candidate_value,
                )
            )
            if profile_adaptive:
                optimizer_step_wall_time = time.perf_counter() - optimizer_started
                optimizer_wall_time += optimizer_step_wall_time
                if attempts == 0:
                    first_optimizer_step_wall_time = optimizer_step_wall_time
                else:
                    steady_optimizer_step_wall_time += optimizer_step_wall_time
            attempts += 1
            objective = objective.record_training_evaluations(term_indices=active_indices)
            control.progress = replace(control.progress, epoch=iteration)
            outcome = TrainingAttemptOutcome(int(outcome))
            if (
                keep_best
                and selection_policy is None
                and outcome != TrainingAttemptOutcome.NONFINITE
            ):
                # The attempt's value is the committed parameters' loss on its
                # frozen realization, so it is a selection candidate as well.
                if control.select(
                    float(attempt_value), params, step=control.progress.update_step
                ):
                    best_loss = float(attempt_value)
            if outcome != TrainingAttemptOutcome.ACCEPTED:
                continue

            update_step = control.progress.update_step + 1
            control.complete_update(update_step)
            params = kernel_state.parameters
            if update is not None:
                pseudo_inverse_steps = update.pseudo_inverse_steps
                term_multipliers = update.term_multipliers
                if update.diagnostic_gradient is not None:
                    previous_gradient = update.diagnostic_gradient
            if training is not None and training.pseudo_transient:
                previous_functions = functions_snapshot
            metrics = kernel_state.rule_state.metrics
            accepted_loss = float(candidate_value)
            selection_evaluation_loss = None
            if (
                keep_best
                and selection_policy is not None
                and selection_policy.due(update_step)
            ):
                selection_prepared = objective.prepare_evaluation(
                    key=functional_site_key(attempt_state, "selection"),
                    iteration=term_iteration,
                )
                selection_evaluation_loss = evaluate_prepared_objective(
                    selection_prepared, eqx.combine(params, held)
                ).total
                improved = control.select(
                    float(selection_evaluation_loss),
                    params,
                    step=update_step,
                    mode=selection_policy.mode,
                    min_delta=selection_policy.min_delta,
                    patience=selection_policy.patience,
                )
                if improved:
                    best_loss = float(selection_evaluation_loss)
            elif keep_best and selection_policy is None:
                if control.select(accepted_loss, params, step=update_step):
                    best_loss = accepted_loss
            elif not keep_best:
                best_loss = accepted_loss
                control.best_payload = params

            log_step = (
                _logging_enabled()
                and int(log_every) > 0
                and update_step % int(log_every) == 0
            )
            tensorboard_step = (
                tensorboard_writer is not None
                and tensorboard_period is not None
                and update_step % int(tensorboard_period) == 0
            )
            session_step = session is not None and update_step % session_every_ == 0
            report_step = log_step or tensorboard_step or session_step
            elapsed = time.perf_counter() - iteration_started
            train_terms = jnp.zeros((len(self.terms),), dtype=jnp.float64)
            train_data_metrics = tuple({} for _ in self.terms)
            eval_terms = jnp.zeros((len(self.evaluation_terms),), dtype=jnp.float64)
            eval_data_metrics = tuple({} for _ in self.evaluation_terms)
            if log_terms and report_step:
                evaluation_functions = eqx.combine(params, held)
                active_values = evaluate_prepared_objective(
                    prepared,
                    evaluation_functions,
                    include_model_losses=False,
                ).term_values
                train_terms = _expanded_train_terms(
                    active_values,
                    active_term_indices=active_indices,
                    num_terms=len(self.terms),
                )
                active_metrics = prepared_data_metrics(
                    prepared,
                    evaluation_functions,
                )
                expanded_metrics = [{} for _ in self.terms]
                for term_index, term_metrics in zip(
                    active_indices,
                    active_metrics,
                    strict=True,
                ):
                    expanded_metrics[term_index] = term_metrics
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
                    iteration=term_iteration,
                )
                eval_terms = evaluate_prepared_objective(
                    prepared_evaluation,
                    evaluation_functions,
                    include_model_losses=False,
                ).term_values
                eval_data_metrics = prepared_data_metrics(
                    prepared_evaluation,
                    evaluation_functions,
                )
            if report_step:
                optimizer_metrics: dict[str, Any] = {
                    "optimizer/kfac/cg_iterations_max": metrics.cg_iterations_max,
                    "optimizer/kfac/cg_relative_residual_max": (
                        metrics.cg_relative_residual_max
                    ),
                    "optimizer/kfac/damping": optim.damping,
                    "optimizer/kfac/factor_condition_estimate_max": (
                        _factor_condition_estimate(
                            kernel_state.rule_state.curvature,
                            damping=optim.damping,
                        )
                    ),
                    "optimizer/kfac/factor_updates": (
                        kernel_state.rule_state.factor_updates
                    ),
                    "optimizer/kfac/line_search_steps": metrics.line_search_steps,
                    "optimizer/kfac/quadratic_update_norm": (
                        metrics.quadratic_update_norm
                    ),
                    "optimizer/kfac/step_size": metrics.accepted_step_size,
                }
                scalars = _training_scalars(
                    loss=accepted_loss,
                    best_loss=best_loss,
                    evaluation_loss=selection_evaluation_loss,
                    iter_time_s=elapsed,
                    train_term_names=term_names,
                    train_terms=train_terms,
                    train_data_metrics=train_data_metrics,
                    train_model_loss_names=(),
                    train_model_loss_terms=jnp.zeros((0,), dtype=jnp.float64),
                    evaluation_term_names=evaluation_term_names,
                    eval_terms=eval_terms,
                    eval_data_metrics=eval_data_metrics,
                    log_terms=log_terms,
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
                        backend="kfac",
                        step=update_step,
                        total_steps=int(num_iter),
                    )
                if tensorboard_step and tensorboard_writer is not None:
                    _write_tensorboard_scalars(
                        tensorboard_writer,
                        scalars,
                        step=update_step,
                    )
                    if update_step % int(tensorboard_flush_every) == 0:
                        tensorboard_writer.flush()
            if (
                training is not None
                and training.checkpoint is not None
                and training.checkpoint.due(update_step)
            ):
                checkpoint_selected = control.selected(params) if keep_best else params
                checkpoint_state = make_training_state(checkpoint_selected)
                checkpoint_solver = replace_solver_state(
                    self,
                    functions=checkpoint_state.best_functions,
                    objective=objective,
                )
                checkpoint_solver = eqx.tree_at(
                    lambda solver: solver.training_state,
                    checkpoint_solver,
                    checkpoint_state,
                    is_leaf=lambda value: value is None,
                )
                publish_checkpoint(checkpoint_solver, checkpoint_state)
            if control.stop_requested:
                break
            if signal_guard.stop_requested:
                _emit_training_signal_stop(
                    "kfac",
                    signal_guard,
                    completed=update_step,
                    total=int(num_iter),
                )
                break

    params = kernel_state.parameters
    rule_state = kernel_state.rule_state
    chosen = control.selected(params) if keep_best else params
    functions = eqx.combine(chosen, held)
    settle_started = time.perf_counter()
    objective = objective.settle(
        functions,
        key=functional_site_key(kernel_state, "settle"),
        iter_=control.progress.epoch + 1,
    )
    if profile_adaptive:
        jax.block_until_ready(objective)
        refresh_wall_time += time.perf_counter() - settle_started
    result = replace_solver_state(
        self,
        functions=functions,
        objective=objective,
    )
    control.emit(
        TrainingIterationKind.RUN_TERMINAL,
        metrics={"completed_steps": control.progress.update_step},
    )
    if training is not None:
        training_state = make_training_state(chosen)
        result = eqx.tree_at(
            lambda solver: solver.training_state,
            result,
            training_state,
            is_leaf=lambda value: value is None,
        )
        if training.checkpoint is not None and training.checkpoint.save_final:
            publish_checkpoint(result, training_state, final=True)
    objective_plane_diagnostics: dict[str, Any] = {}
    if update is not None:
        objective_plane_diagnostics = {
            "objective/physical": update.physical_values(functions).total,
            "objective/surrogate": update.surrogate_loss(chosen, held),
            "gradient_alignment/intra": update.intra_gradient_alignment,
            "gradient_alignment/inter": update.inter_gradient_alignment,
        }
    metrics = rule_state.metrics
    diagnostics = frozendict(
        {
            "profile_enabled": jnp.asarray(profile_adaptive),
            "refresh_wall_time_seconds": jnp.asarray(refresh_wall_time),
            "optimizer_wall_time_seconds": jnp.asarray(optimizer_wall_time),
            "optimizer/kfac/first_step_wall_time_seconds": jnp.asarray(
                first_optimizer_step_wall_time
            ),
            "optimizer/kfac/steady_step_wall_time_seconds": jnp.asarray(
                steady_optimizer_step_wall_time / max(attempts - 1, 1)
            ),
            "optimizer/kfac/step_size": metrics.accepted_step_size,
            "optimizer/kfac/factor_updates": rule_state.factor_updates,
            "optimizer/kfac/cg_iterations_max": metrics.cg_iterations_max,
            "optimizer/kfac/cg_relative_residual_max": metrics.cg_relative_residual_max,
            "optimizer/kfac/quadratic_update_norm": metrics.quadratic_update_norm,
            "optimizer/kfac/factor_condition_estimate_max": (
                _factor_condition_estimate(rule_state.curvature, damping=optim.damping)
            ),
            "optimizer/kfac/damping": jnp.asarray(optim.damping),
            "optimizer/kfac/line_search_steps": metrics.line_search_steps,
            "optimizer/kfac/finite_rejections": kernel_state.finite_rejections,
            "optimizer/kfac/nonfinite_rejections": kernel_state.nonfinite_rejections,
            "optimizer/kfac/num_parameters": jnp.asarray(plan.layout.parameter_count),
            "optimizer/kfac/num_affine_blocks": jnp.asarray(
                len(plan.layout.affine_blocks)
            ),
            "optimizer/kfac/factor_chunk_size": jnp.asarray(optim.factor_chunk_size),
            "optimizer/kfac/jit_requested": jnp.asarray(bool(jit)),
        }
        | objective_plane_diagnostics
        | _functional_ntk_diagnostic_values(latest_ntk_diagnostics)
    )
    return eqx.tree_at(lambda solver: solver.training_diagnostics, result, diagnostics)


__all__ = ["solve_kfac"]
