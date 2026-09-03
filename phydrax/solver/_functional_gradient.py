#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from .._frozendict import frozendict
from .._trainable import combine_trainable, partition_trainable
from .._training import (
    DelayedTargetPolicy,
    EvaluationParametersFn,
    ExponentialMovingAverageTargetPolicy,
    log_training_signal_stop as _log_training_signal_stop,
    resolve_evaluation_parameters,
    TargetParameterState,
    tensorboard_every as _tensorboard_every,
    TensorBoardLogger as _TensorBoardLogger,
    TrainingController,
    TrainingProgress,
    TrainingSignalGuard as _TrainingSignalGuard,
)
from .._training_objective import (
    _GradientAccumulationState,
    _ObjectiveAccumulator,
    _ObjectiveContribution,
)
from ..nn.parameters import ParameterSubspace
from ..nn.parameters._low_rank import validate_low_rank_subspace
from ..optim._composite import CompositeLeastSquaresProblem
from ..optim._iterative import (
    AbstractCompositeLeastSquaresMethod,
    AbstractLeastSquaresMethod,
    AbstractScalarIterativeMethod,
    OptimizationStatus,
    OptimizationTermination,
)
from ..optim._least_squares import LeastSquaresState
from ..optim._mirror_descent import AbstractMirrorOptimizer
from ..optim._riemannian import (
    AbstractRiemannianLineSearchOptimizer,
    AbstractRiemannianOptimizer,
)
from ..optim._scalar import ScalarIterativeState
from ._functional_checkpoint import (
    load_functional_training_checkpoint,
    save_functional_training_checkpoint,
)
from ._functional_objective import (
    evaluate_prepared_objective,
    evaluate_prepared_scalar_remainder,
    prepared_data_metrics,
)
from ._functional_precision import FunctionalPrecisionPolicy
from ._functional_reporting import (
    best_display_value as _best_display_value,
    log_tensorboard_scalars as _log_tensorboard_scalars,
    metric_suffix as _metric_suffix,
    term_label as _term_label,
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
    log_every: int = 1,
    log_terms: bool = True,
    log_path: str | Path | None = None,
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
    accumulation_steps = int(gradient_accumulation)
    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation must be positive.")
    if num_iter == 0:
        return self

    if isinstance(optim, str):
        raise TypeError(
            "optim must be a Phydrax mirror or Riemannian optimizer, or an Optax "
            "transformation, not a string."
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

    log_ctx = (
        open(Path(log_path), "w", encoding="utf-8")
        if log_path is not None
        else nullcontext(None)
    )

    tb_ctx = (
        _TensorBoardLogger(tensorboard_log_dir)
        if tensorboard_log_dir is not None
        else nullcontext(None)
    )

    with log_ctx as log_fp, tb_ctx as tb_writer, _TrainingSignalGuard() as signal_guard:
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
        surrogate_filter_spec = None
        if parameter_paths is None:
            params, non_trainable = partition_trainable(source_functions)
            explicit_subspace = False
        else:
            subspace = ParameterSubspace.from_leaf_paths(
                source_functions,
                parameter_paths,
                alias_groups=parameter_alias_groups,
            )
            if subspace.leaf_shapes != parameter_shapes:
                raise ValueError("FunctionalSolver parameter-subspace shapes changed.")
            if subspace.leaf_dtypes != parameter_dtypes:
                raise ValueError("FunctionalSolver parameter-subspace dtypes changed.")
            validate_low_rank_subspace(source_functions, subspace)
            params, non_trainable = subspace.initial, subspace
            coordinate_paths = parameter_paths + tuple(
                alias for group in parameter_alias_groups for alias in group[1:]
            )
            surrogate_subspace = ParameterSubspace.from_leaf_paths(
                source_functions,
                coordinate_paths,
            )
            surrogate_filter_spec = jax.tree.map(
                lambda leaf: leaf is not None,
                surrogate_subspace.initial,
                is_leaf=lambda leaf: leaf is None,
            )
            explicit_subspace = True
        sharding_policy = None if training is None else training.sharding
        if sharding_policy is not None:
            params = sharding_policy.place_parameters(params)
            non_trainable = sharding_policy.place_tree(non_trainable)

        def reconstruct_functions(current, fixed):
            if explicit_subspace:
                if not isinstance(fixed, ParameterSubspace):
                    raise TypeError("Explicit functional subspace state is invalid.")
                return fixed.reconstruct(current)
            return combine_trainable(current, fixed)

        def surrogate_coordinates(current, fixed):
            if not explicit_subspace:
                return current, fixed
            if surrogate_filter_spec is None:
                raise TypeError("Explicit surrogate coordinate state is invalid.")
            functions = reconstruct_functions(current, fixed)
            return eqx.partition(functions, surrogate_filter_spec)

        line_search_state_types = (
            optax.ScaleByBacktrackingLinesearchState,
            optax.ScaleByZoomLinesearchState,
        )
        preinitialized_opt_state = None
        if _opt_linesearch is not None:
            preinitialized_opt_state = _opt_linesearch.init(params)
            state_leaves = jax.tree.leaves(
                preinitialized_opt_state,
                is_leaf=lambda value: isinstance(value, line_search_state_types),
            )
            if not any(
                isinstance(value, line_search_state_types) for value in state_leaves
            ):
                _opt_standard = _opt_linesearch
                _opt_linesearch = None
        if precision is not None and _opt_standard is None:
            raise ValueError(
                "Functional precision currently supports standard Optax transforms only."
            )
        if accumulation_steps > 1 and _opt_standard is None:
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
            leaf.dtype for leaf in jax.tree.leaves(params) if eqx.is_inexact_array(leaf)
        }
        if precision is not None and len(trainable_dtypes) != 1:
            raise ValueError(
                "Functional precision requires one uniform trainable parameter dtype."
            )
        precision_dtype = None if precision is None else next(iter(trainable_dtypes))
        accumulation_dtypes = tuple(
            jnp.real(leaf).dtype
            for leaf in jax.tree.leaves(params)
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
        if _opt_least_squares is not None and model_loss_names:
            raise ValueError(
                "Least-squares FunctionalSolver methods require a pure "
                "ResidualPenalty objective without model-level scalar losses."
            )
        evaluation_term_names = tuple(_term_label(c) for c in self.evaluation_terms)
        term_sample_size = _train_term_sample_size(
            train_term_sample_size,
            num_terms=len(self.terms),
        )

        def _precision_context():
            return (
                nullcontext()
                if precision is None
                else jax.default_matmul_precision(precision.matmul_precision)
            )

        def _physical_prepared(prepared_):
            return (
                prepared_.physical
                if isinstance(prepared_, PreparedFunctionalUpdate)
                else prepared_
            )

        def _loss_wrt_params(params_, non_trainable_, prepared_):
            if isinstance(prepared_, PreparedFunctionalUpdate):
                surrogate_params, surrogate_non_trainable = surrogate_coordinates(
                    params_,
                    non_trainable_,
                )
                values = prepared_.surrogate_values(
                    surrogate_params,
                    surrogate_non_trainable,
                )
                return values.total, values.flat_values
            functions = reconstruct_functions(params_, non_trainable_)
            with _precision_context():
                values = evaluate_prepared_objective(prepared_, functions)
            return values.total, values.flat_values

        def _term_values_wrt_params(params_, non_trainable_, prepared_):
            functions = reconstruct_functions(params_, non_trainable_)
            with _precision_context():
                return evaluate_prepared_objective(
                    _physical_prepared(prepared_),
                    functions,
                    include_model_losses=False,
                ).term_values

        def _data_metrics_wrt_terms(params_, non_trainable_, prepared_):
            functions = reconstruct_functions(params_, non_trainable_)
            with _precision_context():
                return prepared_data_metrics(_physical_prepared(prepared_), functions)

        loss_fn = eqx.filter_value_and_grad(_loss_wrt_params, has_aux=True)

        is_composite = _opt_composite is not None
        is_least_squares = _opt_least_squares is not None
        is_iterative = _opt_iterative is not None
        iterative_termination = OptimizationTermination(maximum_steps=int(num_iter))
        is_linesearch = _opt_linesearch is not None
        is_mirror = _opt_mirror is not None
        is_riemannian = _opt_riemannian is not None
        is_riemannian_linesearch = isinstance(
            _opt_riemannian, AbstractRiemannianLineSearchOptimizer
        )

        def optax_linesearch_accepted(optimizer_state) -> bool:
            states = tuple(
                value
                for value in jax.tree.leaves(
                    optimizer_state,
                    is_leaf=lambda value: isinstance(value, line_search_state_types),
                )
                if isinstance(value, line_search_state_types)
            )
            if not states:
                raise RuntimeError("Optax line-search state is missing.")
            return all(
                bool(jnp.isfinite(state.learning_rate) & (state.learning_rate > 0.0))
                for state in states
            )

        def solve_step_terms(
            params_,
            non_trainable_,
            opt_state,
            prepared_,
        ):
            if is_composite:
                physical_ = _physical_prepared(prepared_)
                assert _opt_composite is not None
                residual_terms = materialize_prepared_residual_terms(physical_)

                def _residual_fn(p, _):
                    if isinstance(prepared_, PreparedFunctionalUpdate):
                        if prepared_.residual is None:
                            raise ValueError(
                                "GeneralizedGaussNewton requires residual roots."
                            )
                        return prepared_.residual.roots(p)
                    pieces = tuple(
                        prepared_term_residual_vector(
                            p,
                            non_trainable_,
                            self.enforcement,
                            term,
                            iteration=physical_.iteration,
                        )
                        for term in residual_terms
                    )
                    if not pieces:
                        raise ValueError(
                            "GeneralizedGaussNewton requires at least one active "
                            "ResidualPenalty."
                        )
                    return jnp.concatenate(pieces, axis=0)

                def _scalar_fn(p, _):
                    functions = reconstruct_functions(p, non_trainable_)
                    return evaluate_prepared_scalar_remainder(
                        physical_,
                        functions,
                    )

                composite_problem = CompositeLeastSquaresProblem(
                    _residual_fn,
                    _scalar_fn,
                    problem_id="functional-solver-composite",
                )
                params_, opt_state, _ = _opt_composite.step(
                    composite_problem,
                    params_,
                    opt_state,
                    termination=iterative_termination,
                    args=None,
                )
                loss_val, terms = _loss_wrt_params(
                    params_,
                    non_trainable_,
                    prepared_,
                )
                return params_, opt_state, loss_val, terms

            if is_least_squares:
                physical_ = _physical_prepared(prepared_)
                assert _opt_least_squares is not None
                residual_terms = materialize_prepared_residual_terms(
                    physical_, require_all=True
                )

                def _residual_fn(p):
                    if isinstance(prepared_, PreparedFunctionalUpdate):
                        if prepared_.residual is None:
                            raise ValueError(
                                "Least-squares FunctionalSolver methods require "
                                "residual roots."
                            )
                        return prepared_.residual.roots(p)
                    pieces = tuple(
                        prepared_term_residual_vector(
                            p,
                            non_trainable_,
                            self.enforcement,
                            term,
                            iteration=physical_.iteration,
                        )
                        for term in residual_terms
                    )
                    if not pieces:
                        raise ValueError(
                            "Least-squares FunctionalSolver methods require at "
                            "least one active ResidualPenalty."
                        )
                    return jnp.concatenate(pieces, axis=0)

                params_, opt_state, _ = _opt_least_squares.step(
                    _residual_fn,
                    params_,
                    opt_state,
                    termination=iterative_termination,
                )
                loss_val, terms = _loss_wrt_params(
                    params_,
                    non_trainable_,
                    prepared_,
                )
                return params_, opt_state, loss_val, terms

            if is_iterative:
                assert _opt_iterative is not None

                def _iterative_value_fn(p):
                    return _loss_wrt_params(
                        p,
                        non_trainable_,
                        prepared_,
                    )[0]

                params_, opt_state, loss_val = _opt_iterative.step(
                    _iterative_value_fn,
                    params_,
                    opt_state,
                    termination=iterative_termination,
                )
                _, terms = _loss_wrt_params(
                    params_,
                    non_trainable_,
                    prepared_,
                )
                return params_, opt_state, loss_val, terms

            if is_mirror:
                (loss_val, terms), grads = loss_fn(
                    params_,
                    non_trainable_,
                    prepared_,
                )
                assert _opt_mirror is not None
                params_, opt_state = _opt_mirror.update(
                    grads,
                    opt_state,
                    params_,
                )
                return params_, opt_state, loss_val, terms

            if is_riemannian:
                (loss_val, terms), grads = loss_fn(
                    params_,
                    non_trainable_,
                    prepared_,
                )
                assert _opt_riemannian is not None
                if is_riemannian_linesearch:
                    assert isinstance(
                        _opt_riemannian,
                        AbstractRiemannianLineSearchOptimizer,
                    )

                    def _riemannian_value_fn(p):
                        return _loss_wrt_params(
                            p,
                            non_trainable_,
                            prepared_,
                        )[0]

                    params_, opt_state = _opt_riemannian.update(
                        grads,
                        opt_state,
                        params_,
                        value=loss_val,
                        value_fn=_riemannian_value_fn,
                    )
                    loss_val, terms = _loss_wrt_params(
                        params_,
                        non_trainable_,
                        prepared_,
                    )
                else:
                    params_, opt_state = _opt_riemannian.update(
                        grads,
                        opt_state,
                        params_,
                    )
                return params_, opt_state, loss_val, terms

            if is_linesearch:
                import jax.tree_util as jtu

                def _value_fn(p):
                    return _loss_wrt_params(
                        p,
                        non_trainable_,
                        prepared_,
                    )[0]

                (value, _term_values0), grads = loss_fn(
                    params_,
                    non_trainable_,
                    prepared_,
                )
                grads = jtu.tree_map(
                    lambda a: (
                        jnp.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                        if eqx.is_inexact_array(a)
                        else a
                    ),
                    grads,
                    is_leaf=eqx.is_inexact_array,
                )
                assert _opt_linesearch is not None
                updates, opt_state = _opt_linesearch.update(
                    grads,
                    opt_state,
                    params_,
                    value=value,
                    grad=grads,
                    value_fn=_value_fn,
                )
                params_ = eqx.apply_updates(params_, updates)
                loss_val, term_values = _loss_wrt_params(
                    params_,
                    non_trainable_,
                    prepared_,
                )
                return params_, opt_state, loss_val, term_values

            (loss_val, term_values), grads = loss_fn(
                params_,
                non_trainable_,
                prepared_,
            )
            assert _opt_standard is not None
            updates, opt_state = _opt_standard.update(grads, opt_state, params_)
            params_ = eqx.apply_updates(params_, updates)
            return params_, opt_state, loss_val, term_values

        solve_step = (
            eqx.filter_jit(solve_step_terms)
            if jit and not is_linesearch
            else solve_step_terms
        )

        run_standard_gradient = eqx.filter_jit(loss_fn) if jit else loss_fn

        def apply_standard_gradient(params_, opt_state_, gradient_):
            if _opt_standard is None:
                raise RuntimeError("Standard Optax accumulation is not configured.")
            updates, next_state = _opt_standard.update(
                gradient_,
                opt_state_,
                params_,
            )
            return eqx.apply_updates(params_, updates), next_state

        run_standard_update = (
            eqx.filter_jit(apply_standard_gradient) if jit else apply_standard_gradient
        )
        selection_loss_fn = (
            eqx.filter_jit(_loss_wrt_params)
            if jit and evaluation_parameters is not None
            else _loss_wrt_params
        )

        opt = (
            _opt_composite
            if is_composite
            else _opt_least_squares
            if is_least_squares
            else _opt_iterative
            if is_iterative
            else _opt_mirror
            if is_mirror
            else _opt_riemannian
            if is_riemannian
            else (_opt_linesearch if is_linesearch else _opt_standard)
        )
        if opt is None:
            raise ValueError("Optimizer is not configured.")
        opt_state = (
            resume_state.optimizer_state
            if resume_state is not None
            else opt.init(params)
            if preinitialized_opt_state is None
            else preinitialized_opt_state
        )
        restored_objective = None
        current_evaluation_params = resolve_evaluation_parameters(
            evaluation_parameters,
            opt_state,
            params,
        )
        initial_target_params = (
            current_evaluation_params
            if isinstance(target_policy, ExponentialMovingAverageTargetPolicy)
            and target_policy.source == "evaluation"
            else params
        )
        target_state = (
            resume_state.target_state
            if resume_state is not None
            else None
            if target_policy is None
            else TargetParameterState.initialize(initial_target_params, target_policy)
        )
        if (
            resume
            and resume_state is None
            and training is not None
            and training.checkpoint is not None
        ):
            state_template = FunctionalTrainingState(
                current_functions=source_functions,
                best_functions=source_functions,
                previous_functions=(
                    source_functions if training.pseudo_transient else None
                ),
                optimizer_state=opt_state,
                target_state=target_state,
                key=jr.key(seed),
                pseudo_inverse_steps=tuple(
                    policy.initial_inverse_step for policy in training.pseudo_transient
                ),
                term_multipliers=jnp.ones(
                    (
                        0
                        if training.term_balance is None
                        else len(training.term_balance.blocks)
                    ),
                    dtype=float,
                ),
                progress=TrainingProgress(),
                run_id=training.plan_id,
                gradient_accumulation=accumulation_steps,
            )
            restored = load_functional_training_checkpoint(
                training.checkpoint.path,
                self,
                state_template,
                training,
            )
            resume_state = restored.state
            restored_objective = restored.objective
            source_functions = resume_state.current_functions
            if parameter_paths is None:
                params, non_trainable = partition_trainable(source_functions)
            else:
                resumed_subspace = ParameterSubspace.from_leaf_paths(
                    source_functions,
                    parameter_paths,
                    alias_groups=parameter_alias_groups,
                )
                if (
                    resumed_subspace.leaf_shapes != parameter_shapes
                    or resumed_subspace.leaf_dtypes != parameter_dtypes
                ):
                    raise ValueError(
                        "Checkpoint parameter subspace changed shape or dtype."
                    )
                validate_low_rank_subspace(source_functions, resumed_subspace)
                params, non_trainable = resumed_subspace.initial, resumed_subspace
            opt_state = resume_state.optimizer_state
            if sharding_policy is not None:
                params = sharding_policy.place_parameters(params)
                non_trainable = sharding_policy.place_tree(non_trainable)
                opt_state = sharding_policy.place_tree(opt_state)
            target_state = resume_state.target_state
            current_evaluation_params = resolve_evaluation_parameters(
                evaluation_parameters,
                opt_state,
                params,
            )
        selection_policy = None if training is None else training.selection
        initial_progress = (
            resume_state.progress
            if resume_state is not None
            else TrainingProgress(
                best_value=None if selection_policy is not None else float("inf")
            )
        )
        control = TrainingController(
            total_steps=int(num_iter),
            key=jr.key(seed) if resume_state is None else resume_state.key,
            progress=initial_progress,
        )
        if resume_state is None:
            control.best_payload = current_evaluation_params
        elif parameter_paths is None:
            control.best_payload = partition_trainable(resume_state.best_functions)[0]
        else:
            control.best_payload = ParameterSubspace.from_leaf_paths(
                resume_state.best_functions,
                parameter_paths,
                alias_groups=parameter_alias_groups,
            ).initial
        objective = self.objective if restored_objective is None else restored_objective
        if keep_best and selection_policy is not None and resume_state is None:
            initial_selection = objective.prepare_evaluation(
                key=jr.fold_in(control.key, 1200),
                iteration=jnp.asarray(0.0),
                evaluation_kwargs=(
                    None
                    if target_state is None
                    else {
                        "target_functions": reconstruct_functions(
                            target_state.target,
                            non_trainable,
                        )
                    }
                ),
            )
            initial_selection_functions = reconstruct_functions(
                current_evaluation_params, non_trainable
            )
            initial_selection_value = evaluate_prepared_objective(
                initial_selection, initial_selection_functions
            ).total
            control.select(
                float(initial_selection_value),
                current_evaluation_params,
                step=0,
                mode=selection_policy.mode,
                min_delta=selection_policy.min_delta,
                patience=selection_policy.patience,
            )
        out_file = log_fp if log_fp is not None else None
        refresh_wall_time = 0.0
        optimizer_wall_time = 0.0
        first_optimizer_step_wall_time = 0.0
        steady_optimizer_step_wall_time = 0.0
        training_started = time.perf_counter()
        latest_ntk_diagnostics = None
        prepared = None
        previous_functions = (
            None
            if training is None or not training.pseudo_transient
            else source_functions
            if resume_state is None or resume_state.previous_functions is None
            else resume_state.previous_functions
        )
        pseudo_inverse_steps = (
            ()
            if training is None
            else resume_state.pseudo_inverse_steps
            if resume_state is not None
            else tuple(
                policy.initial_inverse_step for policy in training.pseudo_transient
            )
        )
        term_multipliers = (
            jnp.zeros((0,), dtype=float)
            if training is None or training.term_balance is None
            else resume_state.term_multipliers
            if resume_state is not None
            else jnp.ones((len(training.term_balance.blocks),), dtype=float)
        )
        previous_gradient = (
            None if resume_state is None else resume_state.previous_gradient
        )

        def make_training_state(
            current_params,
            selected_params,
            optimizer_state,
        ):
            if training is None:
                raise RuntimeError("Functional training state requires a training plan.")
            current_functions_ = reconstruct_functions(current_params, non_trainable)
            selected_functions_ = reconstruct_functions(selected_params, non_trainable)
            pseudo_inverse_steps_ = pseudo_inverse_steps
            term_multipliers_ = term_multipliers
            return FunctionalTrainingState(
                current_functions=current_functions_,
                best_functions=selected_functions_,
                previous_functions=previous_functions,
                optimizer_state=optimizer_state,
                target_state=target_state,
                key=control.key,
                pseudo_inverse_steps=pseudo_inverse_steps_,
                term_multipliers=term_multipliers_,
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

        start_update_step = (
            0 if resume_state is None else resume_state.progress.update_step
        )
        start_epoch = 0 if resume_state is None else resume_state.progress.epoch

        def publish_checkpoint(checkpoint_solver, checkpoint_state):
            if training is None or training.checkpoint is None:
                return
            if sharding_policy is not None:
                sharding_policy.synchronize(
                    f"functional-checkpoint-before-{checkpoint_state.progress.update_step}"
                )
            if sharding_policy is None or sharding_policy.is_primary_process:
                save_functional_training_checkpoint(
                    training.checkpoint.path,
                    checkpoint_solver,
                    checkpoint_state,
                    training,
                )
            if sharding_policy is not None:
                sharding_policy.synchronize(
                    f"functional-checkpoint-after-{checkpoint_state.progress.update_step}"
                )

        def prepare_microstep(current_objective, current_params, subkey, iteration):
            functions_snapshot_ = reconstruct_functions(
                current_params,
                non_trainable,
            )
            refresh_started_ = time.perf_counter() if profile_adaptive else 0.0
            refreshed = current_objective.refresh(
                functions_snapshot_,
                key=jr.fold_in(subkey, 101),
                iter_=int(iteration),
            )
            refresh_elapsed = 0.0
            if profile_adaptive:
                jax.block_until_ready(refreshed)
                refresh_elapsed = time.perf_counter() - refresh_started_
            _, active_indices, term_scale = _active_train_terms(
                refreshed.terms,
                sample_size=term_sample_size,
                key=jr.fold_in(subkey, 17),
            )
            physical = refreshed.prepare_training(
                active_indices,
                scale=term_scale,
                evaluation_key=subkey,
                sampling_key=jr.fold_in(subkey, 211),
                iteration=jnp.asarray(iteration, dtype=float),
                evaluation_kwargs=(
                    None
                    if target_state is None
                    else {
                        "target_functions": reconstruct_functions(
                            target_state.target,
                            non_trainable,
                        )
                    }
                ),
            )
            if sharding_policy is not None:
                physical = sharding_policy.place_prepared(physical)
            if training is None:
                prepared_ = physical
            else:
                surrogate_params, surrogate_non_trainable = surrogate_coordinates(
                    current_params,
                    non_trainable,
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

        if start_epoch >= int(num_iter):
            resumed_result = replace_solver_state(
                self,
                functions=resume_state.best_functions,
                objective=objective,
            )
            return eqx.tree_at(
                lambda solver: solver.training_state,
                resumed_result,
                resume_state,
                is_leaf=lambda value: value is None,
            )
        for epoch in range(start_epoch, int(num_iter)):
            if signal_guard.stop_requested:
                _log_training_signal_stop(
                    optimizer_label,
                    signal_guard,
                    completed=epoch,
                    total=int(num_iter),
                    file=out_file,
                )
                break
            completed = epoch
            try:
                iter_start = time.perf_counter()
                iter_ = jnp.asarray(epoch + 1, dtype=float)
                if accumulation_steps == 1:
                    subkey = control.split_key()
                    (
                        objective,
                        prepared,
                        active_term_indices,
                        functions_snapshot,
                        refresh_elapsed,
                    ) = prepare_microstep(
                        objective,
                        params,
                        subkey,
                        epoch + 1,
                    )
                    refresh_wall_time += refresh_elapsed
                    if isinstance(prepared, PreparedFunctionalUpdate):
                        pseudo_inverse_steps = prepared.pseudo_inverse_steps
                        term_multipliers = prepared.term_multipliers
                        if prepared.diagnostic_gradient is not None:
                            previous_gradient = prepared.diagnostic_gradient
                    if (
                        isinstance(prepared, PreparedFunctionalUpdate)
                        and training is not None
                        and training.diagnostics is not None
                        and training.diagnostics.ntk
                        and training.diagnostics.due(epoch + 1)
                    ):
                        if prepared.residual is None:
                            raise ValueError("NTK diagnostics require residual roots.")
                        surrogate_params, _ = surrogate_coordinates(
                            params,
                            non_trainable,
                        )
                        latest_ntk_diagnostics = _functional_ntk_diagnostics(
                            prepared.residual,
                            surrogate_params,
                            training.diagnostics,
                            jr.fold_in(subkey, 1701),
                        )
                    pre_update_params = params
                    optimizer_started = time.perf_counter() if profile_adaptive else 0.0
                    params, opt_state, loss_val, term_values = solve_step(
                        params,
                        non_trainable,
                        opt_state,
                        prepared,
                    )
                    control.progress = replace(
                        control.progress,
                        microstep=control.progress.microstep + 1,
                    )
                    if training is not None and training.pseudo_transient:
                        previous_functions = functions_snapshot
                    iterative_step_metrics = (
                        _opt_composite.step_metrics(opt_state)
                        if _opt_composite is not None
                        else _opt_least_squares.step_metrics(opt_state)
                        if _opt_least_squares is not None
                        else _opt_iterative.step_metrics(opt_state)
                        if _opt_iterative is not None
                        else None
                    )
                    riemannian_linesearch_metrics = (
                        _opt_riemannian.step_metrics(opt_state)
                        if is_riemannian_linesearch and _opt_riemannian is not None
                        else None
                    )
                    accepted_update = (
                        bool(iterative_step_metrics.accepted)
                        if iterative_step_metrics is not None
                        else bool(riemannian_linesearch_metrics.line_search_accepted)
                        if riemannian_linesearch_metrics is not None
                        else optax_linesearch_accepted(opt_state)
                        if is_linesearch
                        else True
                    )
                    training_evaluation_multiplier = (
                        1
                        if iterative_step_metrics is None
                        else 2 + int(iterative_step_metrics.globalization_evaluations)
                    )
                    if profile_adaptive:
                        jax.block_until_ready((params, opt_state, loss_val))
                        optimizer_step_wall_time = time.perf_counter() - optimizer_started
                    objective = objective.record_training_evaluations(
                        multiplier=training_evaluation_multiplier,
                        term_indices=active_term_indices,
                    )
                    values_arr = jnp.asarray(term_values, dtype=float)
                    active_term_count = len(prepared.terms)
                    train_term_values = _expanded_train_terms(
                        values_arr[:active_term_count],
                        active_term_indices=active_term_indices,
                        num_terms=len(term_names),
                    )
                    train_model_loss_terms = values_arr[active_term_count:]
                else:
                    pre_update_params = params
                    gradient_accumulator = _GradientAccumulationState.empty(
                        params,
                        accumulation_dtype=accumulation_dtype,
                    )
                    loss_accumulator = _ObjectiveAccumulator()
                    term_accumulators = [_ObjectiveAccumulator() for _ in term_names]
                    model_loss_accumulators = [
                        _ObjectiveAccumulator() for _ in model_loss_names
                    ]
                    optimizer_started = time.perf_counter() if profile_adaptive else 0.0
                    window_refresh_elapsed = 0.0
                    for _ in range(accumulation_steps):
                        subkey = control.split_key()
                        (
                            objective,
                            prepared,
                            active_term_indices,
                            functions_snapshot,
                            refresh_elapsed,
                        ) = prepare_microstep(
                            objective,
                            pre_update_params,
                            subkey,
                            epoch + 1,
                        )
                        refresh_wall_time += refresh_elapsed
                        window_refresh_elapsed += refresh_elapsed
                        (micro_loss, micro_values), micro_gradient = (
                            run_standard_gradient(
                                pre_update_params,
                                non_trainable,
                                prepared,
                            )
                        )
                        micro_contribution = _ObjectiveContribution(
                            micro_loss,
                            jnp.ones((), dtype=jnp.asarray(micro_loss).real.dtype),
                        )
                        gradient_accumulator = gradient_accumulator.add(
                            micro_gradient,
                            micro_contribution,
                        )
                        loss_accumulator = loss_accumulator.add(micro_contribution)
                        values_arr = jnp.asarray(micro_values, dtype=float)
                        active_term_count = len(prepared.terms)
                        for local_index, term_index in enumerate(active_term_indices):
                            term_accumulators[term_index] = term_accumulators[
                                term_index
                            ].add(
                                _ObjectiveContribution(
                                    values_arr[local_index],
                                    jnp.asarray(1.0, dtype=values_arr.dtype),
                                )
                            )
                        for model_index, value in enumerate(
                            values_arr[active_term_count:]
                        ):
                            model_loss_accumulators[model_index] = (
                                model_loss_accumulators[model_index].add(
                                    _ObjectiveContribution(
                                        value,
                                        jnp.asarray(1.0, dtype=values_arr.dtype),
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
                    averaged_gradient = gradient_accumulator.normalized_gradient(
                        pre_update_params
                    )
                    params, opt_state = run_standard_update(
                        pre_update_params,
                        opt_state,
                        averaged_gradient,
                    )
                    loss_val = loss_accumulator.value
                    train_term_values = jnp.asarray(
                        tuple(
                            jnp.nan if accumulator.is_empty else accumulator.value
                            for accumulator in term_accumulators
                        ),
                        dtype=float,
                    )
                    train_model_loss_terms = jnp.asarray(
                        tuple(
                            accumulator.value for accumulator in model_loss_accumulators
                        ),
                        dtype=float,
                    )
                    iterative_step_metrics = None
                    riemannian_linesearch_metrics = None
                    accepted_update = True
                    if profile_adaptive:
                        jax.block_until_ready((params, opt_state, loss_val))
                        optimizer_step_wall_time = max(
                            0.0,
                            time.perf_counter()
                            - optimizer_started
                            - window_refresh_elapsed,
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
                current_evaluation_params = resolve_evaluation_parameters(
                    evaluation_parameters,
                    opt_state,
                    params,
                )
                if target_state is not None:
                    target_state = target_state.update(
                        params,
                        accepted=accepted_update,
                        evaluation_parameters=current_evaluation_params,
                    )
                if accepted_update_hook is not None and accepted_update:
                    accepted_update_hook(accepted_step, current_evaluation_params)
                step = attempt_step
                evaluation_loss = None
                if accepted_update and keep_best and selection_policy is not None:
                    if selection_policy.due(accepted_step):
                        selection_prepared = objective.prepare_evaluation(
                            key=jr.fold_in(subkey, 1201),
                            iteration=iter_,
                        )
                        selection_functions = reconstruct_functions(
                            current_evaluation_params, non_trainable
                        )
                        evaluation_loss = evaluate_prepared_objective(
                            selection_prepared, selection_functions
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
                            params
                            if (
                                is_least_squares
                                or is_iterative
                                or is_linesearch
                                or is_riemannian_linesearch
                            )
                            else pre_update_params
                        )
                        selection_loss = loss_val
                    else:
                        evaluation_loss, _ = selection_loss_fn(
                            current_evaluation_params,
                            non_trainable,
                            prepared,
                        )
                        selection_parameters = current_evaluation_params
                        selection_loss = evaluation_loss
                    control.select(
                        float(selection_loss),
                        selection_parameters,
                        step=accepted_step,
                    )
                if (
                    accepted_update
                    and training is not None
                    and training.checkpoint is not None
                    and training.checkpoint.due(accepted_step)
                ):
                    checkpoint_selected = (
                        control.selected(current_evaluation_params)
                        if keep_best
                        else current_evaluation_params
                    )
                    checkpoint_state = make_training_state(
                        params,
                        checkpoint_selected,
                        opt_state,
                    )
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
                iter_time_s = time.perf_counter() - iter_start
                if signal_guard.stop_requested:
                    _log_training_signal_stop(
                        optimizer_label,
                        signal_guard,
                        completed=step,
                        total=int(num_iter),
                        file=out_file,
                    )
                    break
                console_step = log_every_ > 0 and (step % log_every_ == 0)
                tensorboard_step = tb_every_ is not None and (step % tb_every_ == 0)
                mirror_step_metrics = None
                mirror_constraint_residual = None
                if _opt_mirror is not None and (console_step or tensorboard_step):
                    mirror_step_metrics = _opt_mirror.step_metrics(opt_state)
                    mirror_constraint_residual = (
                        _opt_mirror.parameter_geometry.maximum_constraint_residual(
                            current_evaluation_params
                        )
                    )
                riemannian_step_metrics = None
                riemannian_constraint_residual = None
                if _opt_riemannian is not None and (console_step or tensorboard_step):
                    riemannian_step_metrics = _opt_riemannian.step_metrics(opt_state)
                    riemannian_constraint_residual = (
                        _opt_riemannian.parameter_geometry.maximum_constraint_residual(
                            current_evaluation_params
                        )
                    )
                train_data_metrics: tuple[dict[str, Any], ...] = tuple(
                    {} for _ in self.terms
                )
                eval_terms = jnp.zeros((0,), dtype=float)
                eval_data_metrics: tuple[dict[str, Any], ...] = tuple(
                    {} for _ in self.evaluation_terms
                )
                if log_terms_ and (console_step or tensorboard_step):
                    active_data_metrics = _data_metrics_wrt_terms(
                        current_evaluation_params,
                        non_trainable,
                        prepared,
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
                        key=jr.fold_in(subkey, 2),
                        iteration=iter_,
                        evaluation_kwargs=(
                            None
                            if target_state is None
                            else {
                                "target_functions": reconstruct_functions(
                                    target_state.target,
                                    non_trainable,
                                )
                            }
                        ),
                    )
                    eval_terms = _term_values_wrt_params(
                        current_evaluation_params,
                        non_trainable,
                        prepared_evaluation,
                    )
                    eval_data_metrics = _data_metrics_wrt_terms(
                        current_evaluation_params,
                        non_trainable,
                        prepared_evaluation,
                    )

                if console_step:
                    loss_f = float(loss_val)
                    best_display = _best_display_value(
                        control.progress.best_value,
                        loss_f,
                        keep_best=keep_best,
                    )
                    evaluation_suffix = (
                        ""
                        if evaluation_loss is None
                        else f" eval_loss={float(evaluation_loss):.6e}"
                    )
                    optimizer_suffix = ""
                    if iterative_step_metrics is not None:
                        optimizer_suffix = (
                            " grad="
                            f"{float(iterative_step_metrics.optimality_norm):.6e}"
                            " step_norm="
                            f"{float(iterative_step_metrics.step_norm):.6e}"
                            " trials="
                            f"{int(iterative_step_metrics.globalization_evaluations)}"
                            " accepted="
                            f"{int(iterative_step_metrics.accepted)}"
                            " status="
                            f"{int(iterative_step_metrics.status)}"
                        )
                    if mirror_step_metrics is not None:
                        if mirror_constraint_residual is None:
                            raise RuntimeError(
                                "Mirror diagnostics require a constraint residual."
                            )
                        optimizer_suffix = (
                            " coordinate_grad="
                            f"{float(mirror_step_metrics.coordinate_gradient_norm):.6e}"
                            " dual_step="
                            f"{float(mirror_step_metrics.dual_displacement_norm):.6e}"
                            " bregman_step="
                            f"{float(mirror_step_metrics.bregman_step):.6e}"
                            " constraint="
                            f"{float(mirror_constraint_residual):.6e}"
                        )
                    if riemannian_step_metrics is not None:
                        if riemannian_constraint_residual is None:
                            raise RuntimeError(
                                "Riemannian diagnostics require a constraint residual."
                            )
                        optimizer_suffix = (
                            " rgrad="
                            f"{float(riemannian_step_metrics.gradient_norm):.6e}"
                            " step_norm="
                            f"{float(riemannian_step_metrics.tangent_step_norm):.6e}"
                            " constraint="
                            f"{float(riemannian_constraint_residual):.6e}"
                        )
                        if (
                            _opt_riemannian is not None
                            and _opt_riemannian.optimizer_id
                            in ("riemannian-momentum", "riemannian-adam")
                        ):
                            optimizer_suffix += (
                                " momentum="
                                f"{float(riemannian_step_metrics.momentum_norm):.6e}"
                            )
                        if (
                            _opt_riemannian is not None
                            and _opt_riemannian.optimizer_id == "riemannian-adam"
                        ):
                            optimizer_suffix += (
                                " adaptive_denom=["
                                f"{float(riemannian_step_metrics.adaptive_denominator_minimum):.6e},"
                                f"{float(riemannian_step_metrics.adaptive_denominator_maximum):.6e}]"
                            )
                        if (
                            _opt_riemannian is not None
                            and _opt_riemannian.optimizer_id
                            in (
                                "riemannian-conjugate-gradient",
                                "riemannian-lbfgs",
                            )
                        ):
                            optimizer_suffix += (
                                " line_search="
                                f"{int(riemannian_step_metrics.line_search_evaluations)}"
                                " accepted="
                                f"{int(riemannian_step_metrics.line_search_accepted)}"
                                " reduction="
                                f"{float(riemannian_step_metrics.line_search_reduction):.6e}"
                            )
                            if (
                                _opt_riemannian.optimizer_id
                                == "riemannian-conjugate-gradient"
                            ):
                                optimizer_suffix += (
                                    f" restarted={int(riemannian_step_metrics.restarted)}"
                                )
                            if _opt_riemannian.optimizer_id == "riemannian-lbfgs":
                                optimizer_suffix += (
                                    " restarted="
                                    f"{int(riemannian_step_metrics.restarted)}"
                                    " pair_accepted="
                                    f"{int(riemannian_step_metrics.pair_accepted)}"
                                )
                    print(
                        f"[phydrax][{optimizer_label}] iter {step}/{int(num_iter)} "
                        f"loss={loss_f:.6e}{evaluation_suffix} "
                        f"best={best_display:.6e} iter_time={iter_time_s:.3f}s"
                        f"{optimizer_suffix}",
                        file=out_file,
                    )
                    if log_terms_:
                        for i, (name, val) in enumerate(
                            zip(
                                term_names,
                                list(map(float, train_term_values)),
                                strict=True,
                            )
                        ):
                            suffix = _metric_suffix(train_data_metrics[i])
                            print(
                                f"  [train {i}] {name}: {val:.6e}{suffix}",
                                file=out_file,
                            )
                        for i, (name, val) in enumerate(
                            zip(
                                model_loss_names,
                                list(map(float, train_model_loss_terms)),
                                strict=True,
                            )
                        ):
                            print(
                                f"  [model {i}] {name}: {val:.6e}",
                                file=out_file,
                            )
                        eval_terms_arr = jnp.asarray(eval_terms, dtype=float)
                        for i, (name, val) in enumerate(
                            zip(
                                evaluation_term_names,
                                list(map(float, eval_terms_arr)),
                                strict=True,
                            )
                        ):
                            suffix = _metric_suffix(eval_data_metrics[i])
                            print(
                                f"  [eval {i}] {name}: {val:.6e}{suffix}",
                                file=out_file,
                            )
                if tensorboard_step and tb_writer is not None:
                    loss_f = float(loss_val)
                    best_display = _best_display_value(
                        control.progress.best_value,
                        loss_f,
                        keep_best=keep_best,
                    )
                    _log_tensorboard_scalars(
                        tb_writer,
                        step=step,
                        loss=loss_val,
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
                    )
                    if iterative_step_metrics is not None:
                        tb_writer.scalar(
                            "optimizer/iterative/optimality_norm",
                            iterative_step_metrics.optimality_norm,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/iterative/step_norm",
                            iterative_step_metrics.step_norm,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/iterative/accepted_step_size",
                            iterative_step_metrics.accepted_step_size,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/iterative/globalization_evaluations",
                            iterative_step_metrics.globalization_evaluations,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/iterative/linear_iterations",
                            iterative_step_metrics.linear_iterations,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/iterative/forcing",
                            iterative_step_metrics.forcing,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/iterative/status",
                            iterative_step_metrics.status,
                            step,
                        )
                    if mirror_step_metrics is not None:
                        tb_writer.scalar(
                            "optimizer/mirror/learning_rate",
                            mirror_step_metrics.learning_rate,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/mirror/coordinate_gradient_norm",
                            mirror_step_metrics.coordinate_gradient_norm,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/mirror/dual_displacement_norm",
                            mirror_step_metrics.dual_displacement_norm,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/mirror/bregman_step",
                            mirror_step_metrics.bregman_step,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/mirror/constraint_residual_max",
                            mirror_constraint_residual,
                            step,
                        )
                    if riemannian_step_metrics is not None:
                        tb_writer.scalar(
                            "optimizer/riemannian/learning_rate",
                            riemannian_step_metrics.learning_rate,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/gradient_norm",
                            riemannian_step_metrics.gradient_norm,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/clipping_scale",
                            riemannian_step_metrics.clipping_scale,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/tangent_step_norm",
                            riemannian_step_metrics.tangent_step_norm,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/momentum_norm",
                            riemannian_step_metrics.momentum_norm,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/constraint_residual_max",
                            riemannian_constraint_residual,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/line_search_evaluations",
                            riemannian_step_metrics.line_search_evaluations,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/line_search_accepted",
                            riemannian_step_metrics.line_search_accepted,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/line_search_reduction",
                            riemannian_step_metrics.line_search_reduction,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/conjugacy_beta",
                            riemannian_step_metrics.conjugacy_beta,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/history_pair_count",
                            riemannian_step_metrics.history_pair_count,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/restarted",
                            riemannian_step_metrics.restarted,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/pair_accepted",
                            riemannian_step_metrics.pair_accepted,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/tangent_residual",
                            riemannian_step_metrics.tangent_residual,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/transported_tangent_residual",
                            riemannian_step_metrics.transported_tangent_residual,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/transport_metric_distortion",
                            riemannian_step_metrics.transport_metric_distortion,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/adaptive_denominator_minimum",
                            riemannian_step_metrics.adaptive_denominator_minimum,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/adaptive_denominator_maximum",
                            riemannian_step_metrics.adaptive_denominator_maximum,
                            step,
                        )
                    if step % tb_flush_every_ == 0:
                        tb_writer.flush()
                if iterative_step_metrics is not None and int(
                    iterative_step_metrics.status
                ) != int(OptimizationStatus.ITERATING):
                    break
            except (KeyboardInterrupt, InterruptedError) as exc:
                signal_guard.request_stop_from_exception(exc)
                _log_training_signal_stop(
                    optimizer_label,
                    signal_guard,
                    completed=completed,
                    total=int(num_iter),
                    file=out_file,
                )
                break

        current_evaluation_params = resolve_evaluation_parameters(
            evaluation_parameters,
            opt_state,
            params,
        )
        chosen = (
            control.selected(current_evaluation_params)
            if keep_best
            else current_evaluation_params
        )
        mirror_diagnostics: dict[str, Any] = {}
        if _opt_mirror is not None:
            mirror_geometry = _opt_mirror.parameter_geometry
            if not bool(mirror_geometry.contains(chosen)):
                raise ValueError(
                    "Returned parameters are outside their declared "
                    "ParameterMirrorGeometry."
                )
            mirror_metrics = _opt_mirror.step_metrics(opt_state)
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
        if _opt_riemannian is not None:
            geometry = _opt_riemannian.parameter_geometry
            if not bool(geometry.contains(chosen)):
                raise ValueError(
                    "Returned parameters are outside their declared ParameterGeometry."
                )
            final_metrics = _opt_riemannian.step_metrics(opt_state)
            riemannian_diagnostics = {
                "optimizer/riemannian/num_manifold_leaves": jnp.asarray(
                    geometry.num_manifold_leaves
                ),
                "optimizer/riemannian/learning_rate": final_metrics.learning_rate,
                "optimizer/riemannian/gradient_norm": final_metrics.gradient_norm,
                "optimizer/riemannian/clipping_scale": final_metrics.clipping_scale,
                "optimizer/riemannian/tangent_step_norm": (
                    final_metrics.tangent_step_norm
                ),
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
                "optimizer/riemannian/history_pair_count": (
                    final_metrics.history_pair_count
                ),
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
        if (
            _opt_composite is not None
            or _opt_least_squares is not None
            or _opt_iterative is not None
        ):
            if _opt_composite is not None:
                final_iterative_metrics = _opt_composite.step_metrics(opt_state)
            elif _opt_least_squares is not None:
                final_iterative_metrics = _opt_least_squares.step_metrics(opt_state)
            else:
                assert _opt_iterative is not None
                final_iterative_metrics = _opt_iterative.step_metrics(opt_state)
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
                "optimizer/iterative/linear_status": (
                    final_iterative_metrics.linear_status
                ),
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
            if _opt_composite is not None or _opt_least_squares is not None:
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
                    "optimizer/iterative/scalar_evaluations": (
                        opt_state.scalar_evaluations
                    ),
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
        functions = reconstruct_functions(chosen, non_trainable)
        settle_started = time.perf_counter() if profile_adaptive else 0.0
        with _precision_context():
            objective = objective.settle(
                functions,
                key=jr.fold_in(control.key, 991),
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
        if training is not None:
            training_state = make_training_state(params, chosen, opt_state)
            result = eqx.tree_at(
                lambda solver: solver.training_state,
                result,
                training_state,
                is_leaf=lambda value: value is None,
            )
            if training.checkpoint is not None and training.checkpoint.save_final:
                publish_checkpoint(result, training_state)
        objective_plane_diagnostics: dict[str, Any] = {}
        if isinstance(prepared, PreparedFunctionalUpdate):
            chosen_surrogate, chosen_surrogate_non_trainable = surrogate_coordinates(
                chosen,
                non_trainable,
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
            }
            | objective_plane_diagnostics
            | _functional_ntk_diagnostic_values(latest_ntk_diagnostics)
            | riemannian_diagnostics
            | mirror_diagnostics
            | iterative_diagnostics
        )
        return eqx.tree_at(lambda s: s.training_diagnostics, result, diagnostics)


__all__ = ["solve_gradient"]
