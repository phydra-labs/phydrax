#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Internal accepted-update training kernel shared by every training frontend.

One kernel owns role preflight, authority admission, objective combination,
semantic RNG addressing, the three-outcome attempt lifecycle, and checkpoint
payloads. Frontends supply objectives, an update rule, and their host loop.
"""

from __future__ import annotations

import dataclasses
import functools
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from enum import IntEnum
from typing import Any, ClassVar, final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array, Key, PyTree

from ._differentiation import (
    authority_admits,
    ComponentAuthority,
    DerivativeRoute,
    ObjectiveKind,
)
from ._fingerprint import canonical_fingerprint
from ._identity import NumericRevision
from ._model._component import AbstractComponentSlot, ComponentBinding
from ._sampling._addressing import derive_key, SampleAddress
from ._strict import StrictModule
from ._trainable import (
    ArrayRole,
    combine_parameters,
    ExplicitFreeze,
    LaneLayout,
    parameter_field,
    partition_parameters,
    require_declared_callables,
    require_parameter_roles,
)
from ._training import (
    DelayedTargetPolicy,
    EvaluationParametersFn,
    ExponentialMovingAverageTargetPolicy,
    resolve_evaluation_parameters,
    TargetParameterState,
    TrainingProgress,
)
from ._training_objective import (
    _GradientAccumulationState,
    _merge_objective_contributions,
    _ObjectiveContribution,
)
from ._tree_math import tree_inner, tree_norm, tree_where


TargetPolicy = DelayedTargetPolicy | ExponentialMovingAverageTargetPolicy
_TRAINING_NAMESPACE = "training"
_CHECKPOINT_KIND = "phydrax-training-kernel-checkpoint"
_KEY_IMPLEMENTATIONS = frozenset({"threefry2x32", "rbg", "unsafe_rbg"})
_MANIFEST_FIELDS = frozenset(
    {
        "kind",
        "checkpoint_id",
        "role_schema_id",
        "objective_identity",
        "rule_id",
        "key_impl",
        "structures",
        "fixed_structure",
        "parameter_revision",
        "accepted_boundary",
        "selection",
        "sharding_identity",
    }
)
_ARRAY_FIELDS = frozenset(
    {
        "parameters",
        "model_state",
        "rule_state",
        "targets",
        "accumulation",
        "pending_model_state",
        "root_key_data",
        "cursors",
        "accepted_boundary",
    }
)


class TrainingAttemptOutcome(IntEnum):
    """Closed outcome of one kernel attempt."""

    ACCEPTED = 0
    REJECTED_FINITE = 1
    NONFINITE = 2


class TrainingRejectionBudgetError(RuntimeError):
    """Consecutive rejected attempts exceeded the kernel's rejection budget.

    `state` is the committed state after the rejected attempt (rolled back, with
    advanced cursors and counters) and `evidence` its attempt evidence, when the
    raising driver supplies them (`run_training_attempt` does).
    """

    def __init__(
        self,
        message: str,
        /,
        *,
        consecutive_rejections: int,
        attempt_cursor: int,
        outcome: TrainingAttemptOutcome,
        state: Any = None,
        evidence: Any = None,
    ):
        super().__init__(message)
        self.consecutive_rejections = consecutive_rejections
        self.attempt_cursor = attempt_cursor
        self.outcome = outcome
        self.state = state
        self.evidence = evidence


def _identifier(value: Any, name: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _index(value: Any, /) -> Array:
    return jnp.asarray(value, dtype=jnp.int32)


# Semantic RNG --------------------------------------------------------------------


def _site_key(
    root: Key[Array, ""],
    objective_id: str,
    site: str,
    cursor_kind: str,
    cursor: int | Array,
    microstep: int | Array,
    lane: int | Array | None,
    /,
) -> Key[Array, ""]:
    role = cursor_kind if lane is None else f"{cursor_kind}-lane"
    address = SampleAddress(
        _TRAINING_NAMESPACE,
        _identifier(objective_id, "objective_id"),
        target=(_identifier(site, "site"),),
        role=role,
    )
    indices = (cursor, microstep) if lane is None else (cursor, microstep, lane)
    return derive_key(root, address, *indices)


def training_site_key(
    root: Key[Array, ""],
    /,
    *,
    objective_id: str,
    site: str,
    attempt: int | Array,
    microstep: int | Array,
    lane: int | Array | None = None,
) -> Key[Array, ""]:
    """Key of one attempt-addressed training site.

    Attempt-addressed randomness (dropout, stochastic estimators) is fresh on
    every attempt, including retries after a rejection. The address is
    `SampleAddress("training", objective_id, target=(site,), role="attempt")`
    (`"attempt-lane"` with a lane) folded with `(attempt, microstep[, lane])`.
    """
    return _site_key(root, objective_id, site, "attempt", attempt, microstep, lane)


def training_accepted_site_key(
    root: Key[Array, ""],
    /,
    *,
    objective_id: str,
    site: str,
    accepted: int | Array,
    microstep: int | Array,
    lane: int | Array | None = None,
) -> Key[Array, ""]:
    """Key of one accepted-update-addressed training site.

    Accepted-addressed randomness (batch and sample selection) repeats across
    rejected attempts, so a rule that rejects a trial re-evaluates the same
    realization. Its address role is `"accepted"` (`"accepted-lane"`), disjoint
    from every attempt-addressed key.
    """
    return _site_key(root, objective_id, site, "accepted", accepted, microstep, lane)


@final
class TrainingKeys(StrictModule):
    """Semantic key source handed to one objective evaluation.

    `lane` is set when the kernel trains per-lane parameters; an objective that
    maps lanes internally passes its own `lane` instead.
    """

    root: Key[Array, ""]
    attempt_cursor: Array
    accepted_cursor: Array
    microstep: Array
    lane: Array | None
    objective_id: str = eqx.field(static=True)

    def _lane(self, lane: int | Array | None, /) -> int | Array | None:
        if lane is not None and self.lane is not None:
            raise ValueError("The kernel already addresses this evaluation's lane.")
        return self.lane if lane is None else lane

    def attempt_key(
        self, site: str, /, *, lane: int | Array | None = None
    ) -> Key[Array, ""]:
        """Key of an attempt-addressed site (fresh on every attempt)."""
        return training_site_key(
            self.root,
            objective_id=self.objective_id,
            site=site,
            attempt=self.attempt_cursor,
            microstep=self.microstep,
            lane=self._lane(lane),
        )

    def accepted_key(
        self, site: str, /, *, lane: int | Array | None = None
    ) -> Key[Array, ""]:
        """Key of an accepted-update-addressed site (stable across rejections)."""
        return training_accepted_site_key(
            self.root,
            objective_id=self.objective_id,
            site=site,
            accepted=self.accepted_cursor,
            microstep=self.microstep,
            lane=self._lane(lane),
        )


# Objectives ----------------------------------------------------------------------


ObjectiveFunction = Callable[
    [Any, Any, Any, Any, TrainingKeys], tuple[_ObjectiveContribution, Any, Any]
]


@final
class KernelObjective(StrictModule):
    """One weighted training objective of the kernel.

    `fn(parameters, model_state, fixed, payload, keys)` returns
    `(_ObjectiveContribution, next_model_state, diagnostics)`: a real scalar
    numerator with stop-gradient support, the MODEL_STATE lane after the
    evaluation (same structure, shapes and dtypes), and an array PyTree of
    diagnostics. The kernel combines objectives as `sum(weight * value)` of
    their individually normalized values; support merging happens only across
    the microbatches of one objective. `kind` and `route` decide which parameter
    authorities the objective may train.
    """

    fn: ObjectiveFunction
    objective_id: str = eqx.field(static=True)
    kind: ObjectiveKind = eqx.field(static=True)
    route: DerivativeRoute = eqx.field(static=True)
    weight: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        objective_id: str,
        kind: ObjectiveKind,
        route: DerivativeRoute,
        fn: ObjectiveFunction,
        weight: float = 1.0,
    ):
        identifier = _identifier(objective_id, "objective_id")
        if not isinstance(kind, ObjectiveKind):
            raise TypeError("kind must be an ObjectiveKind.")
        if not isinstance(route, DerivativeRoute):
            raise TypeError("route must be a DerivativeRoute.")
        if route is DerivativeRoute.STOPPED:
            raise ValueError(
                f"Objective {identifier!r} declares the stopped route, which claims "
                "no derivative and cannot train."
            )
        if not callable(fn):
            raise TypeError("fn must be callable.")
        if isinstance(weight, (bool, np.bool_)) or not isinstance(
            weight, (int, float, np.floating, np.integer)
        ):
            raise TypeError("weight must be a real number.")
        weight_ = float(weight)
        if not np.isfinite(weight_) or weight_ <= 0.0:
            raise ValueError("weight must be finite and positive.")
        self.fn = fn
        self.objective_id = identifier
        self.kind = kind
        self.route = route
        self.weight = weight_


# Update rules --------------------------------------------------------------------


@final
class KernelUpdateContext(StrictModule):
    """What an update rule may consult while proposing one attempt.

    `objective_value(parameters)` evaluates the weighted objective at trial
    parameters on the attempt's payload, keys, and committed model state;
    `diagnostics` are the objectives' diagnostics at the current parameters.
    `payload`, `model_state` (committed), and `fixed` are populated only for
    rules that declare `reads_payload` (every rule that forms its own
    derivatives does), and are `None` otherwise. The kernel admits such rules
    only when every objective admits every parameter group, since anything
    built from the raw payload bypasses the per-objective admission mask.
    """

    objective_value: Callable[[Any], Array]
    diagnostics: tuple[Any, ...]
    attempt_cursor: Array
    accepted_cursor: Array
    payload: Any = None
    model_state: Any = None
    fixed: Any = None


class AbstractKernelUpdateRule(StrictModule):
    """Update rule of the training kernel.

    `propose(parameters, gradients, value, rule_state, context)` returns
    `(candidate_parameters, candidate_rule_state, finite_rejection_rule_state,
    accepted)`. The kernel commits the candidate on acceptance. On a finite
    rejection it commits only the rule-state fields named by the class's
    `rejection_commit_policy` (taken from `finite_rejection_rule_state`); an
    empty policy commits nothing. A nonempty policy requires a dataclass rule
    state (a `StrictModule`) that has those fields.
    """

    rejection_commit_policy: eqx.AbstractClassVar[tuple[str, ...]]
    rule_id: eqx.AbstractVar[str]

    @property
    def supports_accumulation(self) -> bool:
        """Whether one proposal may consume several accumulated microbatches."""
        return False

    @property
    def forms_own_derivatives(self) -> bool:
        """Whether the rule forms its own derivatives from the attempt payload.

        Such a rule (native least-squares, iterative, or residual methods)
        receives the payload, committed model state, and FIXED lane on its
        context and a zero gradient: the kernel evaluates the objectives once
        without differentiation, so value, support, next model state, and
        diagnostics stay real while `TrainingAttemptEvidence.gradient_norm` is
        zero. It cannot accumulate microbatches, and the kernel admits it only
        when every objective admits every parameter group (an own derivative
        cannot apply the per-objective admission mask).
        """
        return False

    @property
    def reads_payload(self) -> bool:
        """Whether the rule reads the raw attempt payload on its context.

        True for rules that form their own derivatives and for rules that judge
        a candidate by their own guard evaluation of the payload. The kernel
        admits such a rule only when every objective admits every parameter
        group.
        """
        return self.forms_own_derivatives

    @abstractmethod
    def init(self, parameters: PyTree[Any], /) -> PyTree[Any]:
        raise NotImplementedError

    @abstractmethod
    def propose(
        self,
        parameters: PyTree[Any],
        gradients: PyTree[Any],
        value: Array,
        rule_state: PyTree[Any],
        context: KernelUpdateContext,
        /,
    ) -> tuple[PyTree[Any], PyTree[Any], PyTree[Any], Array]:
        raise NotImplementedError

    def evaluation_parameters(
        self, rule_state: PyTree[Any], parameters: PyTree[Any], /
    ) -> PyTree[Any]:
        """Rule-prescribed evaluation view of the parameters (identity by default)."""
        del rule_state
        return parameters

    def rule_state_finite(self, rule_state: PyTree[Any], /) -> Array:
        """Whether a candidate or rejection rule state is fit to commit.

        Every inexact leaf must be finite by default. Rules whose state carries
        documented not-a-number sentinels (for example unset diagnostic metrics
        of native methods) narrow the check to the numeric state they commit.
        """
        return _finite(rule_state)


@final
class OptaxUpdateRule(AbstractKernelUpdateRule):
    """Optax transformation that accepts every finite update.

    Extra-argument transformations receive the attempt `value` and `grad`.
    `reevaluates_objective=True` also passes `value_fn` (the attempt's
    `objective_value`, as Optax line searches require) and therefore refuses
    microbatch accumulation; such an attempt is accepted only when every Optax
    line-search state reports a finite positive learning rate.
    `evaluation_parameters(optimizer_state, parameters)` is the
    optimizer-prescribed evaluation view (for example schedule-free averaging);
    it drives EMA targets with `source="evaluation"`, and the
    frontend folds its identity into `rule_id`. Nothing commits on a finite
    rejection.
    """

    rejection_commit_policy: ClassVar[tuple[str, ...]] = ()
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    evaluation_view: EvaluationParametersFn | None = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)
    reevaluates_objective: bool = eqx.field(static=True)

    def __init__(
        self,
        optimizer: optax.GradientTransformation,
        /,
        *,
        rule_id: str,
        reevaluates_objective: bool = False,
        evaluation_parameters: EvaluationParametersFn | None = None,
    ):
        if not isinstance(optimizer, optax.GradientTransformation):
            raise TypeError("optimizer must be an optax.GradientTransformation.")
        if not isinstance(reevaluates_objective, bool):
            raise TypeError("reevaluates_objective must be a bool.")
        if reevaluates_objective and not isinstance(
            optimizer, optax.GradientTransformationExtraArgs
        ):
            raise ValueError(
                "Only extra-argument Optax transformations can re-evaluate the objective."
            )
        if evaluation_parameters is not None and not callable(evaluation_parameters):
            raise TypeError("evaluation_parameters must be callable or None.")
        self.optimizer = optimizer
        self.evaluation_view = evaluation_parameters
        self.rule_id = _identifier(rule_id, "rule_id")
        self.reevaluates_objective = reevaluates_objective

    @property
    def supports_accumulation(self) -> bool:
        return not self.reevaluates_objective

    def init(self, parameters: PyTree[Any], /) -> PyTree[Any]:
        return self.optimizer.init(parameters)

    def evaluation_parameters(
        self, rule_state: PyTree[Any], parameters: PyTree[Any], /
    ) -> PyTree[Any]:
        return resolve_evaluation_parameters(self.evaluation_view, rule_state, parameters)

    def propose(
        self,
        parameters: PyTree[Any],
        gradients: PyTree[Any],
        value: Array,
        rule_state: PyTree[Any],
        context: KernelUpdateContext,
        /,
    ) -> tuple[PyTree[Any], PyTree[Any], PyTree[Any], Array]:
        if self.reevaluates_objective:
            updates, next_state = self.optimizer.update(
                gradients,
                rule_state,
                parameters,
                value=value,
                grad=gradients,
                value_fn=context.objective_value,
            )
            candidate = optax.apply_updates(parameters, updates)
            return candidate, next_state, rule_state, _line_search_accepted(next_state)
        elif isinstance(self.optimizer, optax.GradientTransformationExtraArgs):
            updates, next_state = self.optimizer.update(
                gradients, rule_state, parameters, value=value, grad=gradients
            )
        else:
            updates, next_state = self.optimizer.update(gradients, rule_state, parameters)
        candidate = optax.apply_updates(parameters, updates)
        return candidate, next_state, rule_state, jnp.asarray(True)


_LINE_SEARCH_STATES = (
    optax.ScaleByBacktrackingLinesearchState,
    optax.ScaleByZoomLinesearchState,
)


def _line_search_accepted(optimizer_state: PyTree[Any], /) -> Array:
    """Every Optax line-search state found a finite positive learning rate."""
    accepted = jnp.asarray(True)
    for state in jax.tree_util.tree_leaves(
        optimizer_state, is_leaf=lambda value: isinstance(value, _LINE_SEARCH_STATES)
    ):
        if isinstance(state, _LINE_SEARCH_STATES):
            rate = state.learning_rate
            accepted = accepted & jnp.all(jnp.isfinite(rate) & (rate > 0.0))
    return accepted


def _real_dtype(parameters: PyTree[Any], /) -> Any:
    leaves = jax.tree_util.tree_leaves(parameters)
    if not leaves:
        raise ValueError("Update rules require at least one parameter leaf.")
    return jnp.result_type(*(jnp.real(leaf).dtype for leaf in leaves))


@final
class LineSearchState(StrictModule):
    """Initial trial step of the next backtracking attempt."""

    step_size: Array


@final
class BacktrackingLineSearchRule(AbstractKernelUpdateRule):
    """Steepest-descent Armijo backtracking with an adaptive initial step.

    Each attempt tries `step_size * shrink**k` for `k < max_trials` along the
    negative gradient and accepts the first finite trial with
    `f(trial) <= f + sufficient_decrease * step * slope`. Acceptance grows the
    next initial step by `growth`; a finite rejection commits the shrunken step.
    """

    rejection_commit_policy: ClassVar[tuple[str, ...]] = ("step_size",)
    initial_step: float = eqx.field(static=True)
    shrink: float = eqx.field(static=True)
    growth: float = eqx.field(static=True)
    sufficient_decrease: float = eqx.field(static=True)
    max_trials: int = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_step: float = 1.0,
        shrink: float = 0.5,
        growth: float = 2.0,
        sufficient_decrease: float = 1e-4,
        max_trials: int = 20,
    ):
        values = (initial_step, shrink, growth, sufficient_decrease)
        if any(
            isinstance(value, (bool, np.bool_)) or not np.isfinite(float(value))
            for value in values
        ):
            raise ValueError("Line-search hyperparameters must be finite reals.")
        if float(initial_step) <= 0.0:
            raise ValueError("initial_step must be positive.")
        if not 0.0 < float(shrink) < 1.0:
            raise ValueError("shrink must lie in (0, 1).")
        if float(growth) < 1.0:
            raise ValueError("growth must be at least one.")
        if not 0.0 < float(sufficient_decrease) < 1.0:
            raise ValueError("sufficient_decrease must lie in (0, 1).")
        if isinstance(max_trials, bool) or not isinstance(max_trials, int):
            raise TypeError("max_trials must be an integer.")
        if max_trials < 1:
            raise ValueError("max_trials must be positive.")
        self.initial_step = float(initial_step)
        self.shrink = float(shrink)
        self.growth = float(growth)
        self.sufficient_decrease = float(sufficient_decrease)
        self.max_trials = max_trials
        self.rule_id = canonical_fingerprint(
            {
                "kind": "backtracking-line-search",
                "initial_step": self.initial_step,
                "shrink": self.shrink,
                "growth": self.growth,
                "sufficient_decrease": self.sufficient_decrease,
                "max_trials": self.max_trials,
            }
        )

    def init(self, parameters: PyTree[Any], /) -> LineSearchState:
        return LineSearchState(
            jnp.asarray(self.initial_step, dtype=_real_dtype(parameters))
        )

    def propose(
        self,
        parameters: PyTree[Any],
        gradients: PyTree[Any],
        value: Array,
        rule_state: LineSearchState,
        context: KernelUpdateContext,
        /,
    ) -> tuple[PyTree[Any], LineSearchState, LineSearchState, Array]:
        slope = -tree_inner(gradients, gradients)

        def descend(step: Array) -> PyTree[Any]:
            return jax.tree.map(
                lambda parameter, gradient: (
                    parameter - step.astype(parameter.dtype) * gradient
                ),
                parameters,
                gradients,
            )

        def sufficient(step: Array) -> Array:
            trial_value = context.objective_value(descend(step))
            return jnp.isfinite(trial_value) & (
                trial_value <= value + self.sufficient_decrease * step * slope
            )

        def continue_search(carry: tuple[Array, Array, Array]) -> Array:
            index, _, satisfied = carry
            return (index + 1 < self.max_trials) & ~satisfied

        def shrink_step(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
            index, step, _ = carry
            shrunk = step * self.shrink
            return index + 1, shrunk, sufficient(shrunk)

        initial = rule_state.step_size
        _, step, accepted = jax.lax.while_loop(
            continue_search,
            shrink_step,
            (_index(0), initial, sufficient(initial)),
        )
        return (
            descend(step),
            LineSearchState(step * self.growth),
            LineSearchState(step * self.shrink),
            accepted,
        )


class AbstractTrialStepRule(AbstractKernelUpdateRule):
    """Trust-region and damping adapter protocol: one trial judged by its ratio.

    `trial_step` returns `(step, predicted_decrease, rule_state)`. `propose`
    evaluates the objective at `parameters + step` on the attempt payload, forms
    `ratio = (value - trial_value) / predicted_decrease`, accepts a finite trial
    with positive predicted decrease and `ratio >= acceptance_ratio`, and asks
    `adapt` for the accepted and the finite-rejection rule states (for example
    Levenberg-Marquardt damping or a trust radius).
    """

    acceptance_ratio: eqx.AbstractVar[float]

    @abstractmethod
    def trial_step(
        self,
        parameters: PyTree[Any],
        gradients: PyTree[Any],
        value: Array,
        rule_state: PyTree[Any],
        context: KernelUpdateContext,
        /,
    ) -> tuple[PyTree[Any], Array, PyTree[Any]]:
        raise NotImplementedError

    @abstractmethod
    def adapt(
        self, rule_state: PyTree[Any], ratio: Array, /, *, accepted: bool
    ) -> PyTree[Any]:
        raise NotImplementedError

    def propose(
        self,
        parameters: PyTree[Any],
        gradients: PyTree[Any],
        value: Array,
        rule_state: PyTree[Any],
        context: KernelUpdateContext,
        /,
    ) -> tuple[PyTree[Any], PyTree[Any], PyTree[Any], Array]:
        step, predicted, state = self.trial_step(
            parameters, gradients, value, rule_state, context
        )
        candidate = jax.tree.map(jnp.add, parameters, step)
        trial_value = context.objective_value(candidate)
        predicted_ = jnp.asarray(predicted, dtype=trial_value.dtype)
        positive = predicted_ > 0.0
        ratio = jnp.where(
            positive,
            (value - trial_value) / jnp.where(positive, predicted_, 1.0),
            -jnp.inf,
        )
        accepted = jnp.isfinite(trial_value) & positive & (ratio >= self.acceptance_ratio)
        return (
            candidate,
            self.adapt(state, ratio, accepted=True),
            self.adapt(state, ratio, accepted=False),
            accepted,
        )


# Trained trees ---------------------------------------------------------------------


@final
class _FrozenSubspace(StrictModule, ExplicitFreeze):
    """Whole `ParameterSubspace` (frozen complement and alias groups), FIXED."""

    subspace: Any


@final
class SubspaceTrainingTree(StrictModule):
    """Kernel tree that trains exactly the leaves of one `ParameterSubspace`.

    `selected` is the PARAMETER lane (the subspace's selection, which keeps the
    model's structure and component slots); the subspace itself is FIXED below an
    `ExplicitFreeze` holder. `model()` rebuilds the full tree, copying canonical
    values to alias leaves; `subspace` is the subspace at the current selection.
    """

    selected: Any = parameter_field()
    complement: _FrozenSubspace

    @classmethod
    def from_subspace(cls, subspace: Any, /) -> SubspaceTrainingTree:
        from .nn.parameters import ParameterSubspace

        if not isinstance(subspace, ParameterSubspace):
            raise TypeError("subspace must be a ParameterSubspace.")
        return cls(subspace.initial, _FrozenSubspace(subspace))

    @property
    def subspace(self) -> Any:
        return eqx.tree_at(
            lambda value: value.initial, self.complement.subspace, self.selected
        )

    def model(self) -> PyTree[Any]:
        return self.complement.subspace.reconstruct(self.selected)


# Kernel state and evidence --------------------------------------------------------


@final
class ObjectiveAccumulation(StrictModule):
    """Open-window microbatch accumulation of one objective.

    Numerator gradients and the value numerator share the log-stable support
    and scale of `_GradientAccumulationState` and `_merge_objective_contributions`.
    """

    gradient_numerator: Any
    value_numerator: Array
    support: Array
    log_scale: Array


@final
class TrainingKernelState(StrictModule):
    """Committed training state and cursors of one kernel run.

    Parameters, model state, rule state, targets, and the accepted cursor move
    only on accepted attempts. The attempt cursor and the rejection counters are
    kernel bookkeeping and advance on every attempt, except that an unsupported
    attempt leaves `consecutive_rejections` unchanged; it still counts in
    `finite_rejections`, and `TrainingAttemptEvidence.supported` is the signal
    that identifies a skip. `pending_model_state` is the
    MODEL_STATE threaded through the open accumulation window. In lane mode every
    leaf carries the lane axis first.
    """

    parameters: Any
    model_state: Any
    rule_state: Any
    targets: TargetParameterState | None
    accumulation: tuple[ObjectiveAccumulation, ...]
    pending_model_state: Any
    root_key: Key[Array, ""]
    attempt_cursor: Array
    accepted_cursor: Array
    microstep: Array
    consecutive_rejections: Array
    finite_rejections: Array
    nonfinite_rejections: Array
    accepted_boundary: Array


@final
class TrainingAttemptEvidence(StrictModule):
    """Device-side outcome and diagnostics of one attempt.

    `gradient_norm` is the Euclidean norm of the combined (weighted, normalized)
    gradient the update rule received; nonfinite on a nonfinite attempt and zero
    for rules that form their own derivatives.
    """

    outcome: Array
    value: Array
    objective_values: tuple[Array, ...]
    rule_accepted: Array
    finite: Array
    supported: Array
    gradient_norm: Array
    diagnostics: tuple[Any, ...]


@final
class TrainingKernelSpec(StrictModule):
    """Frontend declaration of one kernel run.

    `rejection_budget` is the number of consecutive rejected attempts allowed;
    the next one raises `TrainingRejectionBudgetError` (0 raises on the first).
    Unsupported attempts (skips) do not count; with budget 0 a run may skip any
    number of zero-support windows but fails on its first supported rejection.
    Skip frontends must choose data by an explicit cursor or attempt-addressed
    keys: a skip does not advance the accepted cursor.
    `target_policy` tracks PARAMETER only. `lane_layout` declares lane-mapped
    leaves of the trained tree; mapping any PARAMETER leaf selects lane mode,
    in which every PARAMETER and MODEL_STATE leaf is lane-mapped, each lane is
    an independent training problem, and payload arrays carry the lane axis first.
    """

    rule: AbstractKernelUpdateRule
    lane_layout: LaneLayout | None
    context: str = eqx.field(static=True)
    rejection_budget: int = eqx.field(static=True)
    target_policy: TargetPolicy | None = eqx.field(static=True)
    accumulation_dtype: str = eqx.field(static=True)

    def __init__(
        self,
        rule: AbstractKernelUpdateRule,
        /,
        *,
        context: str,
        rejection_budget: int,
        target_policy: TargetPolicy | None = None,
        lane_layout: LaneLayout | None = None,
        accumulation_dtype: Any = jnp.float64,
    ):
        if not isinstance(rule, AbstractKernelUpdateRule):
            raise TypeError("rule must be an AbstractKernelUpdateRule.")
        context_ = _identifier(context, "context")
        if isinstance(rejection_budget, bool) or not isinstance(rejection_budget, int):
            raise TypeError("rejection_budget must be an integer.")
        if rejection_budget < 0:
            raise ValueError("rejection_budget must be nonnegative.")
        if target_policy is not None and not isinstance(
            target_policy,
            (DelayedTargetPolicy, ExponentialMovingAverageTargetPolicy),
        ):
            raise TypeError("target_policy must be a target parameter policy.")
        if lane_layout is not None and not isinstance(lane_layout, LaneLayout):
            raise TypeError("lane_layout must be a LaneLayout.")
        dtype = jnp.dtype(accumulation_dtype)
        if not jnp.issubdtype(dtype, jnp.floating):
            raise ValueError("accumulation_dtype must be a real floating dtype.")
        self.rule = rule
        self.lane_layout = lane_layout
        self.context = context_
        self.rejection_budget = rejection_budget
        self.target_policy = target_policy
        self.accumulation_dtype = dtype.name


# Tree helpers --------------------------------------------------------------------


def _finite(tree: PyTree[Any], /) -> Array:
    finite = jnp.asarray(True)
    for leaf in jax.tree_util.tree_leaves(tree):
        if eqx.is_inexact_array(leaf):
            finite = finite & jnp.all(jnp.isfinite(leaf))
    return finite


def _require_congruent(expected: PyTree[Any], actual: PyTree[Any], name: str, /) -> None:
    expected_leaves, expected_def = jax.tree_util.tree_flatten(expected)
    actual_leaves, actual_def = jax.tree_util.tree_flatten(actual)
    if expected_def != actual_def:
        raise ValueError(f"{name} must preserve the PyTree structure.")
    for reference, value in zip(expected_leaves, actual_leaves, strict=True):
        if eqx.is_array(reference) != eqx.is_array(value) or (
            eqx.is_array(reference)
            and (reference.shape != value.shape or reference.dtype != value.dtype)
        ):
            raise ValueError(f"{name} must preserve every leaf shape and dtype.")


def _stop_unadmitted(parameters: PyTree[Any], admitted: tuple[bool, ...], /) -> Any:
    if all(admitted):
        return parameters
    leaves, treedef = jax.tree_util.tree_flatten(parameters)
    return jax.tree_util.tree_unflatten(
        treedef,
        [
            leaf if trained else jax.lax.stop_gradient(leaf)
            for leaf, trained in zip(leaves, admitted, strict=True)
        ],
    )


def _add_normalized(
    total: Array, numerator: Array, /, *, support: Array, weight: float
) -> Array:
    """`total + weight * numerator / support`, exactly zero for zero support."""
    positive = support > 0.0
    normalized = jnp.where(
        positive, numerator / jnp.where(positive, support, 1.0), 0.0
    ).astype(total.dtype)
    return total + jnp.asarray(weight, dtype=total.dtype) * normalized


def _authorized_rejection_state(
    policy: tuple[str, ...], source: Any, rejection: Any, /
) -> Any:
    if not policy:
        return source
    return eqx.tree_at(
        lambda state: tuple(getattr(state, name) for name in policy),
        source,
        tuple(getattr(rejection, name) for name in policy),
        is_leaf=lambda value: value is None,
    )


def _require_rejection_fields(rule: AbstractKernelUpdateRule, state: Any, /) -> None:
    policy = rule.rejection_commit_policy
    if not isinstance(policy, tuple) or any(not isinstance(name, str) for name in policy):
        raise TypeError(f"{type(rule).__name__}.rejection_commit_policy must be a tuple.")
    if not policy:
        return
    if not dataclasses.is_dataclass(state):
        raise TypeError(
            f"{type(rule).__name__} commits {policy!r} on rejection, so its rule "
            "state must be a StrictModule with those fields."
        )
    missing = sorted(set(policy) - {field.name for field in dataclasses.fields(state)})
    if missing:
        raise ValueError(
            f"{type(rule).__name__} rejection_commit_policy names unknown rule-state "
            f"fields {missing!r}."
        )


def _leaf_records(tree: PyTree[Any], /) -> dict[str, Any]:
    return {
        jax.tree_util.keystr(path) or "<root>": leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
    }


def _structure_signature(tree: PyTree[Any], /) -> list[dict[str, Any]]:
    """Path, shape, and dtype of every array leaf (concrete or abstract)."""
    return [
        {
            "path": jax.tree_util.keystr(path) or "<root>",
            "shape": list(leaf.shape),
            "dtype": np.dtype(leaf.dtype).str,
        }
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
        if isinstance(leaf, (jax.Array, np.ndarray, jax.ShapeDtypeStruct))
    ]


# Authority resolution --------------------------------------------------------------


def _is_component_owner(node: Any, /) -> bool:
    return isinstance(node, (AbstractComponentSlot, ComponentBinding))


def _owner_authority(node: Any, /) -> ComponentAuthority:
    if isinstance(node, ComponentBinding):
        return node.authority
    return type(node).component_authority


def _component_authorities(
    tree: PyTree[Any], /
) -> tuple[dict[str, ComponentAuthority | None], tuple[tuple[str, str], ...]]:
    """Nearest owning slot or binding authority of every leaf, and bindings.

    Every `ComponentBinding` is re-validated through its contract; its
    `(path, bound_semantic_id)` is recorded.
    """

    authorities: dict[str, ComponentAuthority | None] = {}
    bindings: list[tuple[str, str]] = []

    def visit(
        node: Any, prefix: tuple[Any, ...], authority: ComponentAuthority | None
    ) -> None:
        entries = jax.tree_util.tree_flatten_with_path(
            node, is_leaf=lambda value: value is not node and _is_component_owner(value)
        )[0]
        for path, child in entries:
            keys = prefix + tuple(path)
            if child is not node and _is_component_owner(child):
                if isinstance(child, ComponentBinding):
                    bindings.append(
                        (
                            jax.tree_util.keystr(keys) or "<root>",
                            child.contract().bound_semantic_id,
                        )
                    )
                visit(child, keys, _owner_authority(child))
            else:
                authorities[jax.tree_util.keystr(keys)] = authority

    root = _owner_authority(tree) if _is_component_owner(tree) else None
    if isinstance(tree, ComponentBinding):
        bindings.append(("<root>", tree.contract().bound_semantic_id))
    visit(tree, (), root)
    return authorities, tuple(bindings)


def _parameter_authorities(
    context: str,
    parameter_paths: tuple[str, ...],
    authorities: Mapping[str, ComponentAuthority | None],
    root_authority: ComponentAuthority | None,
    /,
) -> tuple[ComponentAuthority, ...]:
    unowned = tuple(path for path in parameter_paths if authorities[path] is None)
    if unowned and root_authority is None:
        raise ValueError(
            f"{context}: parameters at {unowned!r} have no owning component slot or "
            "ComponentBinding and the frontend declared no root authority."
        )
    return tuple(authorities[path] or root_authority for path in parameter_paths)


def _objective_admission(
    context: str,
    objectives: tuple[KernelObjective, ...],
    parameter_paths: tuple[str, ...],
    parameter_authorities: tuple[ComponentAuthority, ...],
    /,
) -> tuple[tuple[bool, ...], ...]:
    groups: dict[ComponentAuthority, list[str]] = {}
    for path, authority in zip(parameter_paths, parameter_authorities, strict=True):
        groups.setdefault(authority, []).append(path)
    declared = [(objective.route.value, objective.kind.value) for objective in objectives]
    for authority, paths in groups.items():
        if not any(
            authority_admits(authority, objective.route, objective.kind)
            for objective in objectives
        ):
            raise ValueError(
                f"{context}: no admissible training signal for {authority.value} "
                f"parameters at {tuple(paths)!r}; objectives declare (route, kind) "
                f"{declared!r}."
            )
    admission = tuple(
        tuple(
            authority_admits(authority, objective.route, objective.kind)
            for authority in parameter_authorities
        )
        for objective in objectives
    )
    for objective, admitted in zip(objectives, admission, strict=True):
        if not any(admitted):
            raise ValueError(
                f"{context}: objective {objective.objective_id!r} "
                f"({objective.route.value}, {objective.kind.value}) is not admitted "
                f"for any parameter authority {sorted(a.value for a in groups)!r}."
            )
    return admission


def _lane_configuration(
    context: str,
    layout: LaneLayout | None,
    tree: PyTree[Any],
    paths: tuple[str, ...],
    roles: tuple[ArrayRole | None, ...],
    fixed: PyTree[Any],
    /,
) -> tuple[bool, tuple[int | None, ...]]:
    if layout is None:
        return False, ()
    layout.lane_size(tree)
    mapped = frozenset(layout.mapped_paths)
    parameters = tuple(
        path
        for path, role in zip(paths, roles, strict=True)
        if role is ArrayRole.PARAMETER
    )
    if not any(path in mapped for path in parameters):
        return False, ()
    unmapped = tuple(
        path
        for path, role in zip(paths, roles, strict=True)
        if role in (ArrayRole.PARAMETER, ArrayRole.MODEL_STATE) and path not in mapped
    )
    if unmapped:
        raise ValueError(
            f"{context}: per-lane training maps every PARAMETER and MODEL_STATE leaf; "
            f"{unmapped!r} are not lane-mapped (train shared parameters in a "
            "separate kernel)."
        )
    fixed_axes = tuple(
        0 if (jax.tree_util.keystr(path) or "<root>") in mapped else None
        for path, _ in jax.tree_util.tree_flatten_with_path(fixed)[0]
    )
    return True, fixed_axes


# Attempt outcome -------------------------------------------------------------------


class _Proposal(NamedTuple):
    """Validated update-rule proposal of one attempt."""

    parameters: Any
    rule_state: Any
    rejection_rule_state: Any
    accepted: Array


def _classify_outcome(
    finite: Array,
    supported: Array,
    proposal: _Proposal,
    rule: AbstractKernelUpdateRule,
    /,
) -> Array:
    """Device-side three-way outcome of one attempt.

    A rule-accepted proposal is accepted only when the evaluation is finite and
    supported and the candidate parameters and rule state are finite (per
    `rule.rule_state_finite`). Every other finite evaluation is a finite
    rejection, provided the rule's rejection state is finite (or unused because
    the attempt carried no support).
    """
    accepted = (
        finite
        & supported
        & proposal.accepted
        & _finite(proposal.parameters)
        & rule.rule_state_finite(proposal.rule_state)
    )
    rejected = (
        finite
        & ~accepted
        & (~supported | rule.rule_state_finite(proposal.rejection_rule_state))
    )
    return jnp.where(
        accepted,
        _index(TrainingAttemptOutcome.ACCEPTED),
        jnp.where(
            rejected,
            _index(TrainingAttemptOutcome.REJECTED_FINITE),
            _index(TrainingAttemptOutcome.NONFINITE),
        ),
    )


# Prepared kernel -----------------------------------------------------------------


@final
class PreparedTrainingKernel(StrictModule):
    """Preflighted objectives, update rule, FIXED lane, and identities of one run.

    Built by `prepare_training_kernel`. `init(tree, key)` starts a run;
    `attempt(state, payload)` and `accumulate(state, payload)` are pure and
    device-side; `run_training_attempt` owns the one host decision per attempt.
    """

    objectives: tuple[KernelObjective, ...]
    rule: AbstractKernelUpdateRule
    fixed: Any
    lane_layout: LaneLayout | None
    treedef: Any = eqx.field(static=True)
    paths: tuple[str, ...] = eqx.field(static=True)
    roles: tuple[ArrayRole, ...] = eqx.field(static=True)
    parameter_paths: tuple[str, ...] = eqx.field(static=True)
    parameter_authorities: tuple[ComponentAuthority, ...] = eqx.field(static=True)
    admission: tuple[tuple[bool, ...], ...] = eqx.field(static=True)
    component_bindings: tuple[tuple[str, str], ...] = eqx.field(static=True)
    lane_parameters: bool = eqx.field(static=True)
    fixed_lane_axes: tuple[int | None, ...] = eqx.field(static=True)
    target_policy: TargetPolicy | None = eqx.field(static=True)
    rejection_budget: int = eqx.field(static=True)
    accumulation_dtype: str = eqx.field(static=True)
    context: str = eqx.field(static=True)
    parameter_signature: str = eqx.field(static=True)
    model_state_signature: str = eqx.field(static=True)
    role_schema_id: str = eqx.field(static=True)
    objective_identity: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    # Run start ---------------------------------------------------------------------

    def _require_prepared_tree(self, tree: PyTree[Any], /) -> tuple[Any, Any]:
        resolution = require_parameter_roles(tree, context=self.context)
        if resolution.treedef != self.treedef or resolution.roles != self.roles:
            raise ValueError(
                f"{self.context}: the tree does not match the prepared role schema."
            )
        parameters, model_state, fixed = partition_parameters(tree)
        prepared = jax.tree_util.tree_leaves(self.fixed)
        given = jax.tree_util.tree_leaves(fixed)
        if len(prepared) != len(given) or any(
            not (
                expected is actual
                or (
                    not eqx.is_array(expected)
                    and not eqx.is_array(actual)
                    and expected == actual
                )
            )
            for expected, actual in zip(prepared, given, strict=True)
        ):
            raise ValueError(
                f"{self.context}: the tree's FIXED leaves are not the prepared FIXED "
                "lane; prepare a kernel for this tree."
            )
        return parameters, model_state

    def _empty_accumulation(
        self, parameters: PyTree[Any], /
    ) -> tuple[ObjectiveAccumulation, ...]:
        empty = _GradientAccumulationState.empty(
            parameters, accumulation_dtype=self.accumulation_dtype
        )
        return tuple(
            ObjectiveAccumulation(
                empty.gradient_numerator,
                jnp.zeros((), dtype=self.accumulation_dtype),
                empty.support,
                empty.log_scale,
            )
            for _ in self.objectives
        )

    def _initial_state(
        self, parameters: Any, model_state: Any, key: Key[Array, ""], /
    ) -> TrainingKernelState:
        rule_state = self.rule.init(parameters)
        _require_rejection_fields(self.rule, rule_state)
        targets = None
        if self.target_policy is not None:
            source = (
                self.rule.evaluation_parameters(rule_state, parameters)
                if isinstance(self.target_policy, ExponentialMovingAverageTargetPolicy)
                and self.target_policy.source == "evaluation"
                else parameters
            )
            targets = TargetParameterState.initialize(source, self.target_policy)
        zero = _index(0)
        return TrainingKernelState(
            parameters=parameters,
            model_state=model_state,
            rule_state=rule_state,
            targets=targets,
            accumulation=self._empty_accumulation(parameters),
            pending_model_state=model_state,
            root_key=key,
            attempt_cursor=zero,
            accepted_cursor=zero,
            microstep=zero,
            consecutive_rejections=zero,
            finite_rejections=zero,
            nonfinite_rejections=zero,
            accepted_boundary=jnp.asarray(True),
        )

    def _states(
        self, parameters: Any, model_state: Any, key: Key[Array, ""], /
    ) -> TrainingKernelState:
        if not self.lane_parameters:
            return self._initial_state(parameters, model_state, key)
        return eqx.filter_vmap(
            self._initial_state, in_axes=(eqx.if_array(0), eqx.if_array(0), None)
        )(parameters, model_state, key)

    def init(self, tree: PyTree[Any], key: Key[Array, ""], /) -> TrainingKernelState:
        """Start a run from the prepared tree and a typed root key."""
        key_ = jnp.asarray(key)
        if not jax.dtypes.issubdtype(key_.dtype, jax.dtypes.prng_key) or key_.shape:
            raise TypeError("key must be one typed JAX PRNG key (jax.random.key).")
        parameters, model_state = self._require_prepared_tree(tree)
        return self._states(parameters, model_state, key_)

    def tree(self, state: TrainingKernelState, /) -> PyTree[Any]:
        """Committed trained tree."""
        return combine_parameters(state.parameters, state.model_state, self.fixed)

    def target_tree(self, state: TrainingKernelState, /) -> PyTree[Any]:
        """Evaluation view `(target parameters, committed model state, fixed)`."""
        if state.targets is None:
            raise ValueError(f"{self.context}: the kernel tracks no target parameters.")
        return combine_parameters(state.targets.target, state.model_state, self.fixed)

    # Objective evaluation -------------------------------------------------------------

    def _numerator(
        self,
        index: int,
        parameters: Any,
        /,
        *,
        model_state: Any,
        fixed: Any,
        payload: Any,
        keys: TrainingKeys,
    ) -> tuple[Array, tuple[Array, Array, Any, Any]]:
        objective = self.objectives[index]
        trained = _stop_unadmitted(parameters, self.admission[index])
        result = objective.fn(trained, model_state, fixed, payload, keys)
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError(
                f"Objective {objective.objective_id!r} must return "
                "(_ObjectiveContribution, next_model_state, diagnostics)."
            )
        contribution, next_model_state, diagnostics = result
        if not isinstance(contribution, _ObjectiveContribution):
            raise TypeError(
                f"Objective {objective.objective_id!r} must return an "
                "_ObjectiveContribution."
            )
        if not jnp.issubdtype(contribution.numerator.dtype, jnp.floating):
            raise TypeError(
                f"Objective {objective.objective_id!r} numerator must be real."
            )
        _require_congruent(
            model_state,
            next_model_state,
            f"Objective {objective.objective_id!r} next_model_state",
        )
        return contribution.numerator, (
            contribution.support,
            contribution.log_scale,
            next_model_state,
            diagnostics,
        )

    def _evaluate(
        self,
        parameters: Any,
        model_state: Any,
        fixed: Any,
        payload: Any,
        state: TrainingKernelState,
        lane: Array | None,
        /,
        *,
        differentiate: bool,
    ) -> tuple[tuple[_ObjectiveContribution, ...], tuple[Any, ...], Any, tuple[Any, ...]]:
        """Evaluate every objective in declaration order, threading model state."""
        contributions: list[_ObjectiveContribution] = []
        gradients: list[Any] = []
        diagnostics: list[Any] = []
        current = model_state
        for index, objective in enumerate(self.objectives):
            keys = TrainingKeys(
                state.root_key,
                state.attempt_cursor,
                state.accepted_cursor,
                state.microstep,
                lane,
                objective.objective_id,
            )
            numerator_fn = functools.partial(
                self._numerator,
                index,
                model_state=current,
                fixed=fixed,
                payload=payload,
                keys=keys,
            )
            if differentiate:
                (numerator, aux), gradient = eqx.filter_value_and_grad(
                    numerator_fn, has_aux=True
                )(parameters)
                gradients.append(gradient)
            else:
                numerator, aux = numerator_fn(parameters)
            support, log_scale, current, diagnostic = aux
            contributions.append(_ObjectiveContribution(numerator, support, log_scale))
            diagnostics.append(diagnostic)
        return tuple(contributions), tuple(gradients), current, tuple(diagnostics)

    def _add(
        self,
        accumulation: ObjectiveAccumulation,
        gradient: Any,
        contribution: _ObjectiveContribution,
        /,
    ) -> ObjectiveAccumulation:
        window = _GradientAccumulationState(
            accumulation.gradient_numerator,
            accumulation.support,
            accumulation.log_scale,
            0,
            self.accumulation_dtype,
        ).add(gradient, contribution)
        value = _merge_objective_contributions(
            _ObjectiveContribution(
                accumulation.value_numerator,
                accumulation.support,
                accumulation.log_scale,
            ),
            contribution,
        )
        return ObjectiveAccumulation(
            window.gradient_numerator,
            value.numerator.astype(self.accumulation_dtype),
            window.support,
            window.log_scale,
        )

    def _weighted_value(
        self, contributions: Sequence[_ObjectiveContribution], /
    ) -> tuple[Array, tuple[Array, ...]]:
        values = tuple(
            _merge_objective_contributions(
                _ObjectiveContribution(
                    jnp.zeros((), dtype=self.accumulation_dtype),
                    jnp.zeros((), dtype=self.accumulation_dtype),
                    jnp.zeros((), dtype=self.accumulation_dtype),
                ),
                contribution,
            ).value.astype(self.accumulation_dtype)
            for contribution in contributions
        )
        return self._total(values), values

    def _total(self, values: Sequence[Array], /) -> Array:
        total = jnp.zeros((), dtype=self.accumulation_dtype)
        for objective, value in zip(self.objectives, values, strict=True):
            total = total + jnp.asarray(objective.weight, dtype=value.dtype) * value
        return total

    def _combined(
        self, accumulation: Sequence[ObjectiveAccumulation], parameters: Any, /
    ) -> tuple[Array, tuple[Array, ...], Any, Array]:
        """Weighted sum of per-objective normalized values and gradients.

        Each objective is normalized by its own support; a zero-support
        objective contributes an exact zero value and gradient.
        """
        values: list[Array] = []
        gradient = jax.tree.map(jnp.zeros_like, parameters)
        supported = jnp.asarray(False)
        for objective, entry in zip(self.objectives, accumulation, strict=True):
            contribution = _ObjectiveContribution(
                entry.value_numerator, entry.support, entry.log_scale
            )
            values.append(contribution.value.astype(self.accumulation_dtype))
            gradient = jax.tree.map(
                functools.partial(
                    _add_normalized,
                    support=entry.support,
                    weight=objective.weight,
                ),
                gradient,
                entry.gradient_numerator,
            )
            supported = supported | (entry.support > 0.0)
        return self._total(values), tuple(values), gradient, supported

    # Attempts ------------------------------------------------------------------------

    def _lane_call(
        self,
        function: Callable[..., Any],
        state: TrainingKernelState,
        payload: Any,
        /,
    ) -> Any:
        if not self.lane_parameters:
            return function(state, self.fixed, payload, None)
        leaves, treedef = jax.tree_util.tree_flatten(self.fixed)
        axes = self.fixed_lane_axes
        shared = [
            None if axis == 0 else leaf for leaf, axis in zip(leaves, axes, strict=True)
        ]
        mapped = [
            leaf if axis == 0 else None for leaf, axis in zip(leaves, axes, strict=True)
        ]

        def lane_function(lane_state, lane_fixed, lane_payload, lane):
            fixed = jax.tree_util.tree_unflatten(
                treedef,
                [
                    value if axis == 0 else base
                    for value, base, axis in zip(lane_fixed, shared, axes, strict=True)
                ],
            )
            return function(lane_state, fixed, lane_payload, lane)

        lanes = jnp.arange(state.attempt_cursor.shape[0], dtype=jnp.int32)
        return eqx.filter_vmap(
            lane_function,
            in_axes=(eqx.if_array(0), eqx.if_array(0), eqx.if_array(0), 0),
        )(state, mapped, payload, lanes)

    def accumulate(
        self, state: TrainingKernelState, payload: Any, /
    ) -> TrainingKernelState:
        """Add one microbatch to the open window at the committed parameters."""
        return self.accumulate_with_diagnostics(state, payload)[0]

    def accumulate_with_diagnostics(
        self, state: TrainingKernelState, payload: Any, /
    ) -> tuple[TrainingKernelState, tuple[Any, ...]]:
        """`accumulate`, also returning the microbatch's objective diagnostics.

        Frontends that report per-microbatch metrics (for example per-term
        losses merged over the window) read them here; the attempt evidence
        carries only the diagnostics of the closing microbatch.
        """
        if not self.rule.supports_accumulation:
            raise ValueError(
                f"{self.context}: update rule {self.rule.rule_id!r} re-evaluates the "
                "objective and does not accumulate microbatches."
            )
        return self._lane_call(self._accumulate_lane, state, payload)

    def _accumulate_lane(
        self,
        state: TrainingKernelState,
        fixed: Any,
        payload: Any,
        lane: Array | None,
        /,
    ) -> tuple[TrainingKernelState, tuple[Any, ...]]:
        contributions, gradients, next_model_state, diagnostics = self._evaluate(
            state.parameters,
            state.pending_model_state,
            fixed,
            payload,
            state,
            lane,
            differentiate=True,
        )
        accumulation = tuple(
            self._add(entry, gradient, contribution)
            for entry, gradient, contribution in zip(
                state.accumulation, gradients, contributions, strict=True
            )
        )
        return (
            dataclasses.replace(
                state,
                accumulation=accumulation,
                pending_model_state=next_model_state,
                microstep=state.microstep + 1,
                accepted_boundary=jnp.asarray(False),
            ),
            diagnostics,
        )

    def attempt(
        self, state: TrainingKernelState, payload: Any, /
    ) -> tuple[TrainingKernelState, TrainingAttemptEvidence]:
        """Close the window with `payload`, propose, and select the outcome state.

        Accepted commits parameters, rule state, model state, targets, and the
        accepted cursor. A finite rejection commits only the rule's authorized
        rejection state and rejection counters. A nonfinite attempt rolls every
        training quantity back. An unsupported finite attempt (every objective
        has zero support) is a skip: it commits nothing and leaves the
        consecutive-rejection count unchanged. The window is consumed by every
        outcome.
        """
        return self._lane_call(self._attempt_lane, state, payload)

    def _attempt_lane(
        self,
        state: TrainingKernelState,
        fixed: Any,
        payload: Any,
        lane: Array | None,
        /,
    ) -> tuple[TrainingKernelState, TrainingAttemptEvidence]:
        own_derivatives = self.rule.forms_own_derivatives
        reads_payload = own_derivatives or self.rule.reads_payload
        contributions, gradients, next_model_state, diagnostics = self._evaluate(
            state.parameters,
            state.pending_model_state,
            fixed,
            payload,
            state,
            lane,
            differentiate=not own_derivatives,
        )
        if own_derivatives:
            zero_gradient = jax.tree.map(jnp.zeros_like, state.parameters)
            gradients = tuple(zero_gradient for _ in contributions)
        accumulation = tuple(
            self._add(entry, gradient, contribution)
            for entry, gradient, contribution in zip(
                state.accumulation, gradients, contributions, strict=True
            )
        )
        value, values, gradient, supported = self._combined(
            accumulation, state.parameters
        )

        def objective_value(trial_parameters: Any) -> Array:
            trial = self._evaluate(
                trial_parameters,
                state.model_state,
                fixed,
                payload,
                state,
                lane,
                differentiate=False,
            )[0]
            return self._weighted_value(trial)[0]

        proposal = self._propose(
            state,
            gradient,
            value,
            KernelUpdateContext(
                objective_value,
                diagnostics,
                state.attempt_cursor,
                state.accepted_cursor,
                payload if reads_payload else None,
                state.model_state if reads_payload else None,
                fixed if reads_payload else None,
            ),
        )
        finite = (
            jnp.isfinite(value)
            & _finite(values)
            & _finite(gradient)
            & _finite(next_model_state)
        )
        outcome = _classify_outcome(finite, supported, proposal, self.rule)
        evidence = TrainingAttemptEvidence(
            outcome=outcome,
            value=value,
            objective_values=values,
            rule_accepted=proposal.accepted,
            finite=finite,
            supported=supported,
            gradient_norm=tree_norm(gradient),
            diagnostics=diagnostics,
        )
        return (
            self._outcome_state(state, outcome, supported, proposal, next_model_state),
            evidence,
        )

    def _propose(
        self,
        state: TrainingKernelState,
        gradient: Any,
        value: Array,
        context: KernelUpdateContext,
        /,
    ) -> _Proposal:
        candidate, candidate_rule, rejection_rule, accepted = self.rule.propose(
            state.parameters, gradient, value, state.rule_state, context
        )
        rule_id = self.rule.rule_id
        _require_congruent(state.parameters, candidate, f"Rule {rule_id!r} candidate")
        _require_congruent(
            state.rule_state, candidate_rule, f"Rule {rule_id!r} candidate state"
        )
        _require_congruent(
            state.rule_state, rejection_rule, f"Rule {rule_id!r} rejection state"
        )
        accepted_ = jnp.asarray(accepted, dtype=jnp.bool_)
        if accepted_.shape != ():
            raise ValueError(f"Rule {rule_id!r} acceptance must be a scalar.")
        return _Proposal(candidate, candidate_rule, rejection_rule, accepted_)

    def _outcome_state(
        self,
        state: TrainingKernelState,
        outcome: Array,
        supported: Array,
        proposal: _Proposal,
        next_model_state: Any,
        /,
    ) -> TrainingKernelState:
        """Select the committed state of one attempt outcome."""
        empty = self._empty_accumulation(state.parameters)
        zero = _index(0)
        one = _index(1)
        targets = (
            None
            if state.targets is None
            else state.targets.update(
                proposal.parameters,
                accepted=True,
                evaluation_parameters=self.rule.evaluation_parameters(
                    proposal.rule_state, proposal.parameters
                ),
            )
        )
        accepted_state = TrainingKernelState(
            parameters=proposal.parameters,
            model_state=next_model_state,
            rule_state=proposal.rule_state,
            targets=targets,
            accumulation=empty,
            pending_model_state=next_model_state,
            root_key=state.root_key,
            attempt_cursor=state.attempt_cursor + one,
            accepted_cursor=state.accepted_cursor + one,
            microstep=zero,
            consecutive_rejections=zero,
            finite_rejections=state.finite_rejections,
            nonfinite_rejections=state.nonfinite_rejections,
            accepted_boundary=jnp.asarray(True),
        )
        rolled_back = dataclasses.replace(
            state,
            accumulation=empty,
            pending_model_state=state.model_state,
            attempt_cursor=state.attempt_cursor + one,
            microstep=zero,
            consecutive_rejections=state.consecutive_rejections + one,
            accepted_boundary=jnp.asarray(False),
        )
        # An unsupported attempt carried no training signal, so the rule judged
        # nothing: none of its rejection state is authorized and the attempt does
        # not count against the consecutive-rejection budget (a skip).
        rejected_state = dataclasses.replace(
            rolled_back,
            rule_state=tree_where(
                supported,
                _authorized_rejection_state(
                    self.rule.rejection_commit_policy,
                    state.rule_state,
                    proposal.rejection_rule_state,
                ),
                state.rule_state,
            ),
            finite_rejections=state.finite_rejections + one,
            consecutive_rejections=jnp.where(
                supported,
                rolled_back.consecutive_rejections,
                state.consecutive_rejections,
            ),
        )
        nonfinite_state = dataclasses.replace(
            rolled_back, nonfinite_rejections=state.nonfinite_rejections + one
        )
        return tree_where(
            outcome == TrainingAttemptOutcome.ACCEPTED,
            accepted_state,
            tree_where(
                outcome == TrainingAttemptOutcome.REJECTED_FINITE,
                rejected_state,
                nonfinite_state,
            ),
        )


def prepare_training_kernel(
    tree: PyTree[Any],
    objectives: Sequence[KernelObjective],
    spec: TrainingKernelSpec,
    /,
    *,
    root_authority: ComponentAuthority | None,
) -> PreparedTrainingKernel:
    """Preflight roles, authorities, admission, lanes, and identities of one run.

    Every PARAMETER leaf takes the authority of its nearest owning
    `AbstractComponentSlot` or `ComponentBinding`, else the frontend's
    `root_authority`. Every authority group must be admitted by at least one
    objective's `(route, kind)`; each objective trains only the groups that
    admit it (other parameters are stop-gradient inside its evaluation).
    """
    if not isinstance(spec, TrainingKernelSpec):
        raise TypeError("spec must be a TrainingKernelSpec.")
    context = spec.context
    objectives_ = tuple(objectives)
    if not objectives_:
        raise ValueError(f"{context}: at least one objective is required.")
    if any(not isinstance(objective, KernelObjective) for objective in objectives_):
        raise TypeError("objectives must be KernelObjective values.")
    identifiers = [objective.objective_id for objective in objectives_]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{context}: objective ids must be unique.")
    for objective in objectives_:
        require_declared_callables(
            objective.fn, context=f"{context} objective {objective.objective_id!r}"
        )
    if root_authority is not None and not isinstance(root_authority, ComponentAuthority):
        raise TypeError("root_authority must be a ComponentAuthority or None.")

    resolution = require_parameter_roles(tree, context=context)
    paths, roles = resolution.paths, resolution.roles
    parameter_paths = tuple(
        path
        for path, role in zip(paths, roles, strict=True)
        if role is ArrayRole.PARAMETER
    )
    if not parameter_paths:
        raise ValueError(f"{context}: training requires at least one PARAMETER leaf.")
    authorities, bindings = _component_authorities(tree)
    parameter_authorities = _parameter_authorities(
        context, parameter_paths, authorities, root_authority
    )
    admission = _objective_admission(
        context, objectives_, parameter_paths, parameter_authorities
    )
    if spec.rule.forms_own_derivatives and spec.rule.supports_accumulation:
        raise ValueError(
            f"{context}: rule {spec.rule.rule_id!r} forms its own derivatives "
            "and cannot accumulate microbatches."
        )
    if (spec.rule.forms_own_derivatives or spec.rule.reads_payload) and not all(
        all(admitted) for admitted in admission
    ):
        raise ValueError(
            f"{context}: rule {spec.rule.rule_id!r} reads the raw attempt payload, "
            "which requires every objective to admit every parameter authority "
            f"{sorted({a.value for a in parameter_authorities})!r}."
        )
    parameters, model_state, fixed = partition_parameters(tree)
    lane_parameters, fixed_lane_axes = _lane_configuration(
        context, spec.lane_layout, tree, paths, roles, fixed
    )

    role_schema_id = canonical_fingerprint(
        {
            "kind": "training-role-schema",
            "paths": list(paths),
            "roles": [role.value for role in roles],
        }
    )
    objective_identity = canonical_fingerprint(
        {
            "kind": "training-objectives",
            "objectives": [
                [o.objective_id, o.kind.value, o.route.value, o.weight]
                for o in objectives_
            ],
        }
    )
    layout = spec.lane_layout
    checkpoint_id = canonical_fingerprint(
        {
            "kind": "training-kernel-checkpoint",
            "role_schema": role_schema_id,
            "objectives": objective_identity,
            "rule": spec.rule.rule_id,
            "authorities": [
                [path, authority.value]
                for path, authority in zip(
                    parameter_paths, parameter_authorities, strict=True
                )
            ],
            "lanes": (
                None
                if layout is None
                else {"kind": layout.kind, "mapped_paths": list(layout.mapped_paths)}
            ),
        }
    )
    return PreparedTrainingKernel(
        objectives=objectives_,
        rule=spec.rule,
        fixed=fixed,
        lane_layout=layout,
        treedef=resolution.treedef,
        paths=paths,
        roles=roles,
        parameter_paths=parameter_paths,
        parameter_authorities=parameter_authorities,
        admission=admission,
        component_bindings=bindings,
        lane_parameters=lane_parameters,
        fixed_lane_axes=fixed_lane_axes,
        target_policy=spec.target_policy,
        rejection_budget=spec.rejection_budget,
        accumulation_dtype=spec.accumulation_dtype,
        context=context,
        parameter_signature=canonical_fingerprint(_structure_signature(parameters)),
        model_state_signature=canonical_fingerprint(_structure_signature(model_state)),
        role_schema_id=role_schema_id,
        objective_identity=objective_identity,
        checkpoint_id=checkpoint_id,
    )


# Host driver -----------------------------------------------------------------------


@eqx.filter_jit
def _compiled_attempt(
    kernel: PreparedTrainingKernel, state: TrainingKernelState, payload: Any
) -> tuple[TrainingKernelState, TrainingAttemptEvidence]:
    return kernel.attempt(state, payload)


def enforce_rejection_budget(
    kernel: PreparedTrainingKernel,
    outcome: np.ndarray,
    consecutive: np.ndarray,
    attempt: np.ndarray,
    /,
    *,
    state: Any = None,
    evidence: Any = None,
) -> None:
    """Raise when any committed lane exceeds the consecutive-rejection budget.

    `state` and `evidence` (the committed rejection state and the attempt's
    evidence) ride on the error so frontends can report the rejected attempt.
    """
    consecutive_ = np.reshape(consecutive, (-1,))
    worst = int(np.argmax(consecutive_))
    if consecutive_[worst] <= kernel.rejection_budget:
        return
    last = TrainingAttemptOutcome(int(np.reshape(outcome, (-1,))[worst]))
    rejections = int(consecutive_[worst])
    cursor = int(np.reshape(attempt, (-1,))[worst])
    raise TrainingRejectionBudgetError(
        f"{kernel.context}: {rejections} consecutive rejected training attempts "
        f"exceed the rejection budget {kernel.rejection_budget}; the last attempt "
        f"(cursor {cursor - 1}) was {last.name.lower().replace('_', ' ')}.",
        consecutive_rejections=rejections,
        attempt_cursor=cursor,
        outcome=last,
        state=state,
        evidence=evidence,
    )


CommittedUpdateHook = Callable[[TrainingKernelState, TrainingAttemptEvidence], None]


def run_training_attempt(
    kernel: PreparedTrainingKernel,
    state: TrainingKernelState,
    payload: Any,
    /,
    *,
    hooks: Sequence[CommittedUpdateHook] = (),
    jit: bool = True,
) -> tuple[TrainingKernelState, TrainingAttemptEvidence]:
    """Run one attempt with its single host decision.

    The attempt is compiled unless `jit=False` (eager debugging frontends). The
    decision reads the outcome and rejection counters once, raises
    `TrainingRejectionBudgetError` past the budget, and runs the committed-update
    `hooks` only when an attempt was accepted.
    """
    if not isinstance(kernel, PreparedTrainingKernel):
        raise TypeError("kernel must be a PreparedTrainingKernel.")
    if not isinstance(state, TrainingKernelState):
        raise TypeError("state must be a TrainingKernelState.")
    attempt_fn = _compiled_attempt if jit else PreparedTrainingKernel.attempt
    next_state, evidence = attempt_fn(kernel, state, payload)
    outcome, consecutive, attempt = jax.device_get(
        (evidence.outcome, next_state.consecutive_rejections, next_state.attempt_cursor)
    )
    enforce_rejection_budget(
        kernel, outcome, consecutive, attempt, state=next_state, evidence=evidence
    )
    if np.any(outcome == TrainingAttemptOutcome.ACCEPTED):
        for hook in hooks:
            hook(next_state, evidence)
    return next_state, evidence


# Checkpoints -----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingCheckpointPayload:
    """Canonical JSON manifest of identities plus the exact array state.

    `arrays` holds parameters, model state, rule state, targets, accumulation,
    root-key data, cursors, and the accepted-boundary marker; the manifest binds
    them to the kernel's identities. There are no version fields: any identity
    or structure mismatch fails closed on restore.
    """

    manifest: Mapping[str, Any]
    arrays: Mapping[str, Any]


@dataclasses.dataclass(frozen=True, slots=True)
class RestoredTrainingCheckpoint:
    state: TrainingKernelState
    selection: TrainingProgress | None


def _checkpoint_arrays(state: TrainingKernelState, /) -> dict[str, Any]:
    return {
        "parameters": state.parameters,
        "model_state": state.model_state,
        "rule_state": state.rule_state,
        "targets": state.targets,
        "accumulation": state.accumulation,
        "pending_model_state": state.pending_model_state,
        "root_key_data": jr.key_data(state.root_key),
        "cursors": {
            "attempt": state.attempt_cursor,
            "accepted": state.accepted_cursor,
            "microstep": state.microstep,
            "consecutive_rejections": state.consecutive_rejections,
            "finite_rejections": state.finite_rejections,
            "nonfinite_rejections": state.nonfinite_rejections,
        },
        "accepted_boundary": state.accepted_boundary,
    }


def _parameter_revision(
    kernel: PreparedTrainingKernel, parameters: Any, /
) -> NumericRevision:
    return NumericRevision(kernel.checkpoint_id, _leaf_records(parameters))


def _expected_structures(
    kernel: PreparedTrainingKernel, arrays: Mapping[str, Any], /
) -> dict[str, Any]:
    template = jax.eval_shape(
        lambda parameters, model_state: _checkpoint_arrays(
            kernel._states(parameters, model_state, jr.key(0))
        ),
        arrays["parameters"],
        arrays["model_state"],
    )
    return {name: _structure_signature(value) for name, value in template.items()}


def build_training_checkpoint(
    kernel: PreparedTrainingKernel,
    state: TrainingKernelState,
    /,
    *,
    selection: TrainingProgress | None = None,
    sharding_identity: str | None = None,
    allow_intermediate: bool = False,
) -> TrainingCheckpointPayload:
    """Build the checkpoint payload of one committed state.

    Only accepted-update boundaries are checkpoint-eligible unless
    `allow_intermediate` records an open window or a post-rejection state; the
    manifest carries the accepted-boundary marker either way.
    """
    if not isinstance(kernel, PreparedTrainingKernel):
        raise TypeError("kernel must be a PreparedTrainingKernel.")
    if not isinstance(state, TrainingKernelState):
        raise TypeError("state must be a TrainingKernelState.")
    if selection is not None and not isinstance(selection, TrainingProgress):
        raise TypeError("selection must be a TrainingProgress or None.")
    sharding = (
        None
        if sharding_identity is None
        else _identifier(sharding_identity, "sharding_identity")
    )
    arrays = _checkpoint_arrays(state)
    boundary = bool(np.all(jax.device_get(state.accepted_boundary)))
    if not boundary and not allow_intermediate:
        raise ValueError(
            f"{kernel.context}: the state is not at an accepted-update boundary."
        )
    manifest = {
        "kind": _CHECKPOINT_KIND,
        "checkpoint_id": kernel.checkpoint_id,
        "role_schema_id": kernel.role_schema_id,
        "objective_identity": kernel.objective_identity,
        "rule_id": kernel.rule.rule_id,
        "key_impl": str(jr.key_impl(state.root_key)),
        "structures": {
            name: _structure_signature(value) for name, value in arrays.items()
        },
        "fixed_structure": _structure_signature(kernel.fixed),
        "parameter_revision": _parameter_revision(kernel, state.parameters).revision_id,
        "accepted_boundary": boundary,
        "selection": None if selection is None else dataclasses.asdict(selection),
        "sharding_identity": sharding,
    }
    return TrainingCheckpointPayload(manifest, arrays)


def _verify_checkpoint(
    kernel: PreparedTrainingKernel,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, Any],
    sharding_identity: str | None,
    /,
) -> None:
    """Fail closed unless the payload is exactly a checkpoint of `kernel`."""
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("Training checkpoint manifest fields do not match.")
    # Component identities are compared before the combined checkpoint identity
    # so a mismatch names its cause.
    expected_identity = {
        "kind": _CHECKPOINT_KIND,
        "role_schema_id": kernel.role_schema_id,
        "objective_identity": kernel.objective_identity,
        "rule_id": kernel.rule.rule_id,
        "checkpoint_id": kernel.checkpoint_id,
        "fixed_structure": _structure_signature(kernel.fixed),
        "sharding_identity": sharding_identity,
    }
    for name, expected in expected_identity.items():
        if manifest[name] != expected:
            raise ValueError(f"Training checkpoint {name} does not match this kernel.")
    if set(arrays) != set(manifest["structures"]) or set(arrays) != _ARRAY_FIELDS:
        raise ValueError("Training checkpoint arrays do not match the manifest.")
    observed = {name: _structure_signature(value) for name, value in arrays.items()}
    if observed != manifest["structures"]:
        raise ValueError(
            "Training checkpoint arrays do not match the manifest structures."
        )
    if (
        canonical_fingerprint(observed["parameters"]) != kernel.parameter_signature
        or canonical_fingerprint(observed["model_state"]) != kernel.model_state_signature
    ):
        raise ValueError(
            "Training checkpoint parameter structure does not match this kernel."
        )
    if observed != _expected_structures(kernel, arrays):
        raise ValueError(
            "Training checkpoint state structure does not match this kernel."
        )
    if manifest["key_impl"] not in _KEY_IMPLEMENTATIONS:
        raise ValueError("Training checkpoint root key implementation is invalid.")
    boundary = bool(np.all(jax.device_get(arrays["accepted_boundary"])))
    if manifest["accepted_boundary"] is not boundary:
        raise ValueError("Training checkpoint accepted-boundary marker is inconsistent.")
    if (
        _parameter_revision(kernel, arrays["parameters"]).revision_id
        != manifest["parameter_revision"]
    ):
        raise ValueError("Training checkpoint parameter revision does not match.")


def restore_training_checkpoint(
    kernel: PreparedTrainingKernel,
    payload: TrainingCheckpointPayload,
    /,
    *,
    sharding_identity: str | None = None,
) -> RestoredTrainingCheckpoint:
    """Verify a checkpoint against `kernel` and rebuild its committed state.

    Manifest fields, identities, structures, the key implementation, the
    accepted-boundary marker, and the parameters' numeric revision must all
    match; anything else raises `ValueError`.
    """
    if not isinstance(kernel, PreparedTrainingKernel):
        raise TypeError("kernel must be a PreparedTrainingKernel.")
    if not isinstance(payload, TrainingCheckpointPayload):
        raise TypeError("payload must be a TrainingCheckpointPayload.")
    manifest, arrays = payload.manifest, payload.arrays
    _verify_checkpoint(kernel, manifest, arrays, sharding_identity)
    key_impl = manifest["key_impl"]
    cursors = arrays["cursors"]
    state = TrainingKernelState(
        parameters=arrays["parameters"],
        model_state=arrays["model_state"],
        rule_state=arrays["rule_state"],
        targets=arrays["targets"],
        accumulation=tuple(arrays["accumulation"]),
        pending_model_state=arrays["pending_model_state"],
        root_key=jr.wrap_key_data(
            jnp.asarray(arrays["root_key_data"], dtype=jnp.uint32), impl=key_impl
        ),
        attempt_cursor=_index(cursors["attempt"]),
        accepted_cursor=_index(cursors["accepted"]),
        microstep=_index(cursors["microstep"]),
        consecutive_rejections=_index(cursors["consecutive_rejections"]),
        finite_rejections=_index(cursors["finite_rejections"]),
        nonfinite_rejections=_index(cursors["nonfinite_rejections"]),
        accepted_boundary=jnp.asarray(arrays["accepted_boundary"], dtype=jnp.bool_),
    )
    selection = manifest["selection"]
    return RestoredTrainingCheckpoint(
        state, None if selection is None else TrainingProgress(**selection)
    )


__all__ = [
    "enforce_rejection_budget",
    "AbstractKernelUpdateRule",
    "AbstractTrialStepRule",
    "BacktrackingLineSearchRule",
    "build_training_checkpoint",
    "CommittedUpdateHook",
    "KernelObjective",
    "KernelUpdateContext",
    "LineSearchState",
    "ObjectiveAccumulation",
    "OptaxUpdateRule",
    "prepare_training_kernel",
    "PreparedTrainingKernel",
    "restore_training_checkpoint",
    "RestoredTrainingCheckpoint",
    "SubspaceTrainingTree",
    "run_training_attempt",
    "training_accepted_site_key",
    "training_site_key",
    "TrainingAttemptEvidence",
    "TrainingAttemptOutcome",
    "TrainingCheckpointPayload",
    "TrainingKernelSpec",
    "TrainingKernelState",
    "TrainingKeys",
    "TrainingRejectionBudgetError",
]
