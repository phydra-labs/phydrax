#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-work sampling model-predictive control for discrete control problems."""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import isfinite, prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from .._bounds import Bounds
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..optim import AbstractRiskMeasure, CVaRRisk, EntropicRisk, MeanVarianceRisk
from ._dynamics import DiscreteControlDynamics
from ._parameterization import AbstractControlParameterization
from ._problem import ControlProblem


SamplingMPCUpdate: TypeAlias = Literal["predictive", "cem"]
SamplingMPCBoundPolicy: TypeAlias = Literal["clip", "reject"]
SamplingMPCAggregation: TypeAlias = Literal["expectation", "worst_case", "risk_measure"]
SamplingMPCWarmStartTerminal: TypeAlias = Literal["hold", "zero"]
SamplingMPCRealizationPolicy: TypeAlias = Literal["fixed", "resample"]
SamplingMPCRealizationBinding: TypeAlias = Callable[[Any, PyTree[Any]], Any]


class SamplingMPCRealizations(StrictModule, NonTrainableState):
    """Fixed-capacity physical model realizations for one sampling MPC plan."""

    parameters: PyTree[Array]
    weights: Array
    support_mask: Array
    realization_ids: tuple[str, ...] = eqx.field(static=True)
    posterior_id: str = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    policy: SamplingMPCRealizationPolicy = eqx.field(static=True)
    count: int = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameters: PyTree[Any],
        realization_ids: tuple[str, ...],
        /,
        *,
        weights: ArrayLike | None = None,
        support_mask: ArrayLike | None = None,
        posterior_id: str,
        campaign_id: str,
        policy: SamplingMPCRealizationPolicy = "fixed",
        batch_id: str | None = None,
    ):
        identifiers = tuple(str(value) for value in realization_ids)
        if not identifiers or any(not value for value in identifiers):
            raise ValueError("realization_ids must be non-empty identifiers.")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("realization_ids must be unique.")
        count = len(identifiers)
        leaves = jax.tree.leaves(parameters)
        if not leaves:
            raise ValueError(
                "realization parameters must contain at least one array leaf."
            )
        converted = jax.tree.map(jnp.asarray, parameters)
        for leaf in jax.tree.leaves(converted):
            if leaf.ndim < 1 or leaf.shape[0] != count:
                raise ValueError(
                    "Every realization parameter leaf must have the common leading "
                    f"capacity axis {(count,)}; got {leaf.shape}."
                )
            if not bool(jnp.all(jnp.isfinite(leaf))):
                raise ValueError("Realization parameter leaves must be finite.")
        distribution, support = _model_distribution(
            weights,
            support_mask,
            (count,),
            np.dtype(jax.tree.leaves(converted)[0].dtype),
        )
        posterior = str(posterior_id)
        campaign = str(campaign_id)
        if not posterior or not campaign:
            raise ValueError("posterior_id and campaign_id must be non-empty.")
        if policy not in ("fixed", "resample"):
            raise ValueError("realization policy must be 'fixed' or 'resample'.")
        identity = (
            canonical_fingerprint(
                {
                    "kind": "sampling-mpc-realizations",
                    "parameters": array_tree_fingerprint(converted),
                    "realization_ids": list(identifiers),
                    "weights": array_tree_fingerprint(distribution),
                    "support": array_tree_fingerprint(support),
                    "posterior": posterior,
                    "campaign": campaign,
                    "policy": policy,
                }
            )
            if batch_id is None
            else str(batch_id)
        )
        if not identity:
            raise ValueError("batch_id must be non-empty.")
        self.parameters = converted
        self.weights = distribution
        self.support_mask = support
        self.realization_ids = identifiers
        self.posterior_id = posterior
        self.campaign_id = campaign
        self.policy = policy
        self.count = count
        self.batch_id = identity


class SamplingMPCStatus(IntEnum):
    """Termination status of one complete fixed-work sampling solve."""

    SUCCESS = 0
    NO_VALID_CANDIDATE = 1


class SamplingMPCPlan(StrictModule, NonTrainableState):
    """Static Gaussian work with an explicit, fixed-capacity model axis.

    Legacy plans use ``ControlProblem.case_shape`` as that axis. Realization-aware
    plans instead require a scalar-case problem and bind one parameter PyTree to
    each model slot without changing the generic control problem API.
    """

    problem: ControlProblem
    parameterization: AbstractControlParameterization
    bounds: Bounds | None
    risk_measure: AbstractRiskMeasure | None
    realizations: SamplingMPCRealizations | None
    realization_binding: SamplingMPCRealizationBinding | None = eqx.field(static=True)
    model_weights: Array
    model_support: Array
    candidate_count: int = eqx.field(static=True)
    iteration_count: int = eqx.field(static=True)
    elite_count: int = eqx.field(static=True)
    update: SamplingMPCUpdate = eqx.field(static=True)
    update_rate: float = eqx.field(static=True)
    minimum_standard_deviation: float = eqx.field(static=True)
    bound_policy: SamplingMPCBoundPolicy = eqx.field(static=True)
    aggregation: SamplingMPCAggregation = eqx.field(static=True)
    warm_start_terminal: SamplingMPCWarmStartTerminal = eqx.field(static=True)
    model_shape: tuple[int, ...] = eqx.field(static=True)
    model_count: int = eqx.field(static=True)
    knot_count: int = eqx.field(static=True)
    parameter_shape: tuple[int, ...] = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)
    sampler_id: str = eqx.field(static=True)
    realization_binding_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def initialize(
        self,
        nominal_controls: ArrayLike,
        standard_deviation: ArrayLike,
        /,
    ) -> SamplingMPCState:
        """Create the array-only distribution state for this plan."""

        return initialize_sampling_mpc(self, nominal_controls, standard_deviation)

    def shift(self, state: SamplingMPCState, /) -> SamplingMPCState:
        """Shift one distribution by one knot under the declared tail policy."""

        return shift_sampling_mpc_state(self, state)

    def solve(
        self,
        state: SamplingMPCState,
        key: Array,
        /,
        *,
        warm_start: ArrayLike = False,
    ) -> SamplingMPCResult:
        """Execute all declared sampling work from caller-owned randomness."""

        return solve_sampling_mpc(self, state, key, warm_start=warm_start)


class SamplingMPCState(StrictModule, NonTrainableState):
    """Gaussian coefficient distribution carried between sampling MPC calls."""

    mean: Array
    standard_deviation: Array
    solve_count: Array
    plan_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)


class SamplingMPCEvidence(StrictModule, NonTrainableState):
    """Complete fixed-capacity sampling, rollout, objective, and update history."""

    candidate_controls: Array
    model_objectives: Array
    model_rollout_valid: Array
    model_feasible: Array
    candidate_objectives: Array
    candidate_accepted: Array
    elite_indices: Array
    elite_accepted: Array
    mean_history: Array
    standard_deviation_history: Array
    realization_indices: Array
    realization_weights: Array
    realization_support: Array
    completed_iterations: Array
    candidate_evaluations: Array
    model_rollouts: Array
    candidate_axis: int = eqx.field(static=True)
    model_axis: int = eqx.field(static=True)
    model_shape: tuple[int, ...] = eqx.field(static=True)
    realization_ids: tuple[str, ...] = eqx.field(static=True)
    posterior_id: str = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    realization_policy: SamplingMPCRealizationPolicy = eqx.field(static=True)
    realization_batch_id: str = eqx.field(static=True)
    realization_binding_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @property
    def control_history(self) -> Array:
        return self.candidate_controls

    @property
    def model_objective_history(self) -> Array:
        return self.model_objectives

    @property
    def objective_history(self) -> Array:
        return self.candidate_objectives

    @property
    def rollout_valid_history(self) -> Array:
        return self.model_rollout_valid

    @property
    def model_valid(self) -> Array:
        return (~self.realization_support) | (
            self.model_rollout_valid & self.model_feasible
        )

    @property
    def candidate_valid(self) -> Array:
        return self.candidate_accepted

    @property
    def elite_valid(self) -> Array:
        return self.elite_accepted


class SamplingMPCSelectedReplay(StrictModule, NonTrainableState):
    """Selected rollout with candidate and accepted discrete transition payloads."""

    states: Array
    controls: Array
    valid: Array
    transition_candidate_states: Array
    transition_accepted_states: Array
    transition_attempted: Array
    transition_successful: Array
    transition_status: Array
    model_objectives: Array
    model_feasible: Array
    realization_indices: Array
    realization_weights: Array
    realization_support: Array
    realization_ids: tuple[str, ...] = eqx.field(static=True)
    posterior_id: str = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    realization_batch_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)

    @property
    def accepted(self) -> Array:
        return jnp.all(
            (~self.realization_support)
            | (
                jnp.all(self.valid, axis=-1)
                & self.model_feasible
                & jnp.all(self.transition_successful, axis=-1)
            )
        )


class SamplingMPCResult(StrictModule, NonTrainableState):
    """Best observed finite-work sample, with no optimality or robustness claim."""

    state: SamplingMPCState
    evidence: SamplingMPCEvidence
    replay: SamplingMPCSelectedReplay
    controls: Array
    action: Array
    objective: Array
    valid: Array
    status: Array
    selected_iteration: Array
    selected_candidate: Array
    completed: Array
    problem_id: str = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(SamplingMPCStatus.SUCCESS))

    @property
    def selected_controls(self) -> Array:
        return self.controls

    @property
    def selected_action(self) -> Array:
        return self.action

    @property
    def total_candidate_evaluations(self) -> Array:
        return self.evidence.candidate_evaluations

    @property
    def total_model_rollouts(self) -> Array:
        return self.evidence.model_rollouts


def _positive_integer(value: int, name: str, /) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _model_distribution(
    weights: ArrayLike | None,
    support_mask: ArrayLike | None,
    model_shape: tuple[int, ...],
    dtype: np.dtype,
    /,
) -> tuple[Array, Array]:
    count = prod(model_shape) if model_shape else 1
    weight_dtype = dtype if np.issubdtype(dtype, np.inexact) else np.dtype(float)
    if support_mask is None:
        support = (
            jnp.ones((count,), dtype=bool)
            if weights is None
            else jnp.asarray(weights).reshape((count,)) > 0.0
        )
    else:
        support_values = jnp.asarray(support_mask, dtype=bool)
        if tuple(support_values.shape) not in ((count,), model_shape):
            raise ValueError(
                "model_support must have flattened model shape "
                f"{(count,)} or declared model shape {model_shape}; "
                f"got {support_values.shape}."
            )
        support = support_values.reshape((count,))
    if not bool(jnp.any(support)):
        raise ValueError("model_support must contain at least one supported model.")
    if weights is None:
        values = support.astype(weight_dtype)
    else:
        weight_values = jnp.asarray(weights, dtype=weight_dtype)
        if tuple(weight_values.shape) not in ((count,), model_shape):
            raise ValueError(
                "model_weights must have flattened model shape "
                f"{(count,)} or declared model shape {model_shape}; "
                f"got {weight_values.shape}."
            )
        values = weight_values.reshape((count,))
        if bool(jnp.any(~jnp.isfinite(values))) or bool(jnp.any(values < 0.0)):
            raise ValueError("model_weights must be finite and non-negative.")
        if bool(jnp.any(jnp.where(support, values <= 0.0, values != 0.0))):
            raise ValueError(
                "Supported models require positive weight and unsupported models "
                "require exactly zero weight."
            )
    total = jnp.sum(values)
    if not bool(total > 0.0):
        raise ValueError("model_weights must have positive supported total weight.")
    return values / total, support


def _risk_identity(risk: AbstractRiskMeasure, /) -> object:
    if isinstance(risk, MeanVarianceRisk):
        parameters: object = {"coefficient": risk.coefficient}
    elif isinstance(risk, CVaRRisk):
        parameters = {"alpha": risk.alpha}
    elif isinstance(risk, EntropicRisk):
        parameters = {"aversion": risk.aversion}
    else:
        parameters = array_tree_fingerprint(risk)
    return {"risk_id": risk.risk_id, "parameters": parameters}


def plan_sampling_mpc(
    problem: ControlProblem,
    parameterization: AbstractControlParameterization,
    /,
    *,
    candidate_count: int,
    iteration_count: int,
    elite_count: int | None = None,
    update: SamplingMPCUpdate = "cem",
    update_rate: float = 1.0,
    minimum_standard_deviation: float = 0.0,
    bounds: Bounds | None = None,
    bound_policy: SamplingMPCBoundPolicy = "clip",
    risk: Literal["expectation", "worst_case"] | AbstractRiskMeasure = "expectation",
    model_weights: ArrayLike | None = None,
    model_support: ArrayLike | None = None,
    realizations: SamplingMPCRealizations | None = None,
    realization_binding: SamplingMPCRealizationBinding | None = None,
    realization_binding_id: str | None = None,
    warm_start_terminal: SamplingMPCWarmStartTerminal = "hold",
) -> SamplingMPCPlan:
    """Plan a fixed number of Gaussian candidate rollouts and CEM updates."""

    if not isinstance(problem, ControlProblem):
        raise TypeError("problem must be a ControlProblem.")
    if not isinstance(problem.dynamics, DiscreteControlDynamics):
        raise ValueError("Sampling MPC currently supports discrete dynamics only.")
    if not isinstance(parameterization, AbstractControlParameterization):
        raise TypeError(
            "parameterization must implement AbstractControlParameterization."
        )
    if parameterization.control_shape != problem.control_shape:
        raise ValueError("parameterization control_shape does not match the problem.")
    parameter_shape = parameterization.parameter_shape
    if len(parameter_shape) < 1 or parameter_shape[1:] != problem.control_shape:
        raise ValueError(
            "Sampling MPC requires one leading knot axis followed by control_shape."
        )
    candidates = _positive_integer(candidate_count, "candidate_count")
    iterations = _positive_integer(iteration_count, "iteration_count")
    elites = (
        candidates
        if elite_count is None
        else _positive_integer(elite_count, "elite_count")
    )
    if elites > candidates:
        raise ValueError("elite_count must not exceed candidate_count.")
    if update not in ("predictive", "cem"):
        raise ValueError("update must be 'predictive' or 'cem'.")
    rate = float(update_rate)
    if not isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError("update_rate must be finite and lie in [0, 1].")
    minimum = float(minimum_standard_deviation)
    if not isfinite(minimum) or minimum < 0.0:
        raise ValueError("minimum_standard_deviation must be finite and non-negative.")
    if bounds is not None and not isinstance(bounds, Bounds):
        raise TypeError("bounds must be a Bounds or None.")
    if bound_policy not in ("clip", "reject"):
        raise ValueError("bound_policy must be 'clip' or 'reject'.")
    if warm_start_terminal not in ("hold", "zero"):
        raise ValueError("warm_start_terminal must be 'hold' or 'zero'.")

    risk_measure: AbstractRiskMeasure | None
    aggregation: SamplingMPCAggregation
    if isinstance(risk, AbstractRiskMeasure):
        risk_measure = risk
        aggregation = "risk_measure"
        risk_identity = _risk_identity(risk)
    elif risk in ("expectation", "worst_case"):
        risk_measure = None
        aggregation = risk
        risk_identity = risk
    else:
        raise TypeError(
            "risk must be 'expectation', 'worst_case', or an AbstractRiskMeasure."
        )

    dtype = np.dtype(problem.initial_state.real.dtype)
    if realizations is None:
        if realization_binding is not None or realization_binding_id is not None:
            raise ValueError(
                "realization_binding is only valid with explicit realizations."
            )
        model_shape = problem.case_shape
        model_count = prod(model_shape) if model_shape else 1
        weights, support = _model_distribution(
            model_weights,
            model_support,
            model_shape,
            dtype,
        )
        binding_id = "sampling-mpc:legacy-case-axis"
    else:
        if not isinstance(realizations, SamplingMPCRealizations):
            raise TypeError("realizations must be SamplingMPCRealizations or None.")
        if problem.case_shape:
            raise ValueError(
                "Explicit realization mode requires a scalar-case ControlProblem; "
                "the realization capacity is the sole model axis."
            )
        if model_weights is not None or model_support is not None:
            raise ValueError(
                "model_weights/model_support are owned by explicit realizations."
            )
        if realization_binding is not None and not callable(realization_binding):
            raise TypeError("realization_binding must be callable or None.")
        if realization_binding_id is None:
            binding_id = (
                "sampling-mpc:replace-problem-args" if realization_binding is None else ""
            )
        else:
            binding_id = str(realization_binding_id)
        if not binding_id:
            raise ValueError(
                "A custom realization_binding requires a non-empty "
                "realization_binding_id."
            )
        model_shape = (realizations.count,)
        model_count = realizations.count
        weights = realizations.weights
        support = realizations.support_mask
    bound_identity: object = None
    if bounds is not None:
        template = jnp.zeros(parameter_shape, dtype=dtype)
        lower, upper = bounds.materialize(template)
        if bool(jnp.any(~jnp.isfinite(lower) & (lower != -jnp.inf))):
            raise ValueError("lower bounds must contain finite values or -inf.")
        if bool(jnp.any(~jnp.isfinite(upper) & (upper != jnp.inf))):
            raise ValueError("upper bounds must contain finite values or inf.")
        bound_identity = array_tree_fingerprint((lower, upper))

    plan_id = canonical_fingerprint(
        {
            "kind": "sampling-mpc-plan",
            "problem": problem.problem_id,
            "parameterization": parameterization.parameterization_id,
            "candidate_count": candidates,
            "iteration_count": iterations,
            "elite_count": elites,
            "update": update,
            "update_rate": rate,
            "minimum_standard_deviation": minimum,
            "bounds": bound_identity,
            "bound_policy": bound_policy,
            "risk": risk_identity,
            "model_weights": array_tree_fingerprint(weights),
            "model_support": array_tree_fingerprint(support),
            "model_shape": list(model_shape),
            "realization_batch": (
                None if realizations is None else realizations.batch_id
            ),
            "realization_binding": binding_id,
            "warm_start_terminal": warm_start_terminal,
        }
    )
    return SamplingMPCPlan(
        problem,
        parameterization,
        bounds,
        risk_measure,
        realizations,
        realization_binding,
        weights,
        support,
        candidates,
        iterations,
        elites,
        update,
        rate,
        minimum,
        bound_policy,
        aggregation,
        warm_start_terminal,
        model_shape,
        model_count,
        parameter_shape[0],
        parameter_shape,
        problem.control_shape,
        problem.problem_id,
        parameterization.parameterization_id,
        "sampling-mpc:predictive-gaussian",
        binding_id,
        plan_id,
    )


def initialize_sampling_mpc(
    plan: SamplingMPCPlan,
    nominal_controls: ArrayLike,
    standard_deviation: ArrayLike,
    /,
) -> SamplingMPCState:
    """Initialize one finite Gaussian proposal without consuming randomness."""

    if not isinstance(plan, SamplingMPCPlan):
        raise TypeError("plan must be a SamplingMPCPlan.")
    mean = jnp.asarray(nominal_controls)
    if tuple(mean.shape) != plan.parameter_shape:
        raise ValueError(
            f"nominal_controls must have shape {plan.parameter_shape}; got {mean.shape}."
        )
    if jnp.issubdtype(mean.dtype, jnp.complexfloating):
        raise TypeError("nominal_controls must be real-valued.")
    if not jnp.issubdtype(mean.dtype, jnp.inexact):
        mean = mean.astype(float)
    deviation = jnp.asarray(standard_deviation, dtype=mean.dtype)
    deviation_shape = tuple(deviation.shape)
    padding = len(plan.parameter_shape) - len(deviation_shape)
    padded_shape = (1,) * max(padding, 0) + deviation_shape
    if padding < 0 or any(
        source not in (1, target)
        for source, target in zip(padded_shape, plan.parameter_shape, strict=True)
    ):
        raise ValueError(
            "standard_deviation must broadcast to the control parameter shape."
        )
    deviation = jnp.broadcast_to(deviation, plan.parameter_shape)
    if bool(jnp.any(~jnp.isfinite(mean))):
        raise ValueError("nominal_controls must be finite.")
    if bool(jnp.any(~jnp.isfinite(deviation))) or bool(jnp.any(deviation < 0.0)):
        raise ValueError("standard_deviation must be finite and non-negative.")
    if plan.bounds is not None and plan.bound_policy == "clip":
        mean = plan.bounds.project(mean)
    return SamplingMPCState(
        mean,
        deviation,
        jnp.asarray(0, dtype=jnp.int32),
        plan.plan_id,
        canonical_fingerprint({"kind": "sampling-mpc-state", "plan": plan.plan_id}),
    )


def _shift_knots(values: Array, terminal: Array, /) -> Array:
    return jnp.concatenate((values[1:], terminal[None]), axis=0)


def shift_sampling_mpc_state(
    plan: SamplingMPCPlan,
    state: SamplingMPCState,
    /,
) -> SamplingMPCState:
    """Shift the proposal horizon exactly once and preserve its uncertainty tail."""

    _validate_state(plan, state)
    terminal_mean = (
        state.mean[-1]
        if plan.warm_start_terminal == "hold"
        else jnp.zeros_like(state.mean[-1])
    )
    return SamplingMPCState(
        _shift_knots(state.mean, terminal_mean),
        _shift_knots(state.standard_deviation, state.standard_deviation[-1]),
        state.solve_count,
        state.plan_id,
        state.state_id,
    )


def _validate_state(plan: SamplingMPCPlan, state: SamplingMPCState, /) -> None:
    if not isinstance(plan, SamplingMPCPlan) or not isinstance(state, SamplingMPCState):
        raise TypeError("plan/state must be SamplingMPCPlan/SamplingMPCState.")
    if state.plan_id != plan.plan_id:
        raise ValueError("Sampling MPC state plan_id does not match the plan.")
    if tuple(state.mean.shape) != plan.parameter_shape:
        raise ValueError("Sampling MPC state mean has the wrong parameter shape.")
    if tuple(state.standard_deviation.shape) != plan.parameter_shape:
        raise ValueError(
            "Sampling MPC state standard_deviation has the wrong parameter shape."
        )


def _candidate_bounds(plan: SamplingMPCPlan, raw: Array, /) -> tuple[Array, Array]:
    finite = jnp.all(jnp.isfinite(raw), axis=tuple(range(1, raw.ndim)))
    if plan.bounds is None:
        return raw, finite
    lower, upper = plan.bounds.materialize(raw[0])
    inside = jnp.all((raw >= lower) & (raw <= upper), axis=tuple(range(1, raw.ndim)))
    if plan.bound_policy == "clip":
        return jnp.clip(raw, lower, upper), finite
    return raw, finite & inside


def _realization_inputs(
    plan: SamplingMPCPlan,
    key: Array,
    solve_count: Array,
    /,
) -> tuple[PyTree[Array] | None, Array, Array, Array]:
    if plan.realizations is None:
        return (
            None,
            plan.model_weights,
            plan.model_support,
            jnp.arange(plan.model_count, dtype=jnp.int32),
        )
    realizations = plan.realizations
    if realizations.policy == "fixed":
        return (
            realizations.parameters,
            realizations.weights,
            realizations.support_mask,
            jnp.arange(realizations.count, dtype=jnp.int32),
        )
    logits = jnp.where(
        realizations.support_mask,
        jnp.log(realizations.weights),
        -jnp.inf,
    )
    indices = jax.lax.stop_gradient(
        jax.random.categorical(
            jax.random.fold_in(
                jax.random.fold_in(key, solve_count),
                jnp.asarray(2**31 - 1, dtype=jnp.uint32),
            ),
            logits,
            shape=(realizations.count,),
        ).astype(jnp.int32)
    )
    parameters = jax.tree.map(
        lambda leaf: jnp.take(leaf, indices, axis=0),
        realizations.parameters,
    )
    weights = jnp.full(
        (realizations.count,),
        1.0 / realizations.count,
        dtype=realizations.weights.dtype,
    )
    return parameters, weights, jnp.ones_like(realizations.support_mask), indices


def _evaluate_models(
    plan: SamplingMPCPlan,
    controls: Array,
    parameters: PyTree[Array] | None,
    /,
):
    if plan.realizations is None:
        coefficients = jnp.broadcast_to(controls, plan.model_shape + plan.parameter_shape)
        return plan.problem.evaluate(plan.parameterization, coefficients)
    if parameters is None:
        raise ValueError("Explicit realization mode requires parameter values.")

    def evaluate_one(realization_parameters):
        args = (
            realization_parameters
            if plan.realization_binding is None
            else plan.realization_binding(
                plan.problem.args,
                realization_parameters,
            )
        )
        bound_problem = eqx.tree_at(
            lambda problem: problem.args,
            plan.problem,
            args,
            is_leaf=lambda value: value is None,
        )
        return bound_problem.evaluate(plan.parameterization, controls)

    return jax.vmap(evaluate_one)(parameters)


def _evaluate_candidate(
    plan: SamplingMPCPlan,
    controls: Array,
    parameters: PyTree[Array] | None,
    /,
) -> tuple[Array, Array, Array]:
    evaluation = _evaluate_models(plan, controls, parameters)
    objectives = evaluation.sampled_loss.total.reshape((plan.model_count,))
    rollout_valid = evaluation.valid.reshape((plan.model_count,)) & jnp.isfinite(
        objectives
    )
    feasible = evaluation.feasibility.feasible.reshape((plan.model_count,))
    return objectives, rollout_valid, feasible


def _realization_metadata(
    plan: SamplingMPCPlan,
    /,
) -> tuple[
    tuple[str, ...],
    str,
    str,
    SamplingMPCRealizationPolicy,
    str,
]:
    if plan.realizations is None:
        return (
            tuple(f"{plan.problem_id}:case:{index}" for index in range(plan.model_count)),
            f"{plan.problem_id}:legacy-posterior",
            f"{plan.problem_id}:legacy-campaign",
            "fixed",
            f"{plan.problem_id}:case-axis",
        )
    return (
        plan.realizations.realization_ids,
        plan.realizations.posterior_id,
        plan.realizations.campaign_id,
        plan.realizations.policy,
        plan.realizations.batch_id,
    )


def _selected_replay(
    plan: SamplingMPCPlan,
    controls: Array,
    parameters: PyTree[Array] | None,
    weights: Array,
    support: Array,
    indices: Array,
    /,
) -> SamplingMPCSelectedReplay:
    evaluation = _evaluate_models(plan, controls, parameters)
    trajectory = evaluation.trajectory
    transition = trajectory.transition_evidence
    if transition is None:
        raise ValueError("Sampling MPC discrete replay requires transition evidence.")
    realization_ids, posterior_id, campaign_id, _, batch_id = _realization_metadata(plan)
    stopped = lambda value: jax.lax.stop_gradient(value)
    return SamplingMPCSelectedReplay(
        stopped(
            trajectory.states.reshape(
                (plan.model_count, plan.problem.time_grid.num_times)
                + plan.problem.state_shape
            )
        ),
        stopped(
            trajectory.controls.reshape(
                (plan.model_count, plan.problem.time_grid.num_steps)
                + plan.problem.control_shape
            )
        ),
        stopped(
            trajectory.valid.reshape((plan.model_count, plan.problem.time_grid.num_times))
        ),
        stopped(
            transition.candidate_states.reshape(
                (plan.model_count, plan.problem.time_grid.num_steps)
                + plan.problem.state_shape
            )
        ),
        stopped(
            transition.accepted_states.reshape(
                (plan.model_count, plan.problem.time_grid.num_steps)
                + plan.problem.state_shape
            )
        ),
        stopped(
            transition.attempted.reshape(
                (plan.model_count, plan.problem.time_grid.num_steps)
            )
        ),
        stopped(
            transition.successful.reshape(
                (plan.model_count, plan.problem.time_grid.num_steps)
            )
        ),
        stopped(
            transition.status.reshape(
                (plan.model_count, plan.problem.time_grid.num_steps)
            )
        ),
        stopped(evaluation.sampled_loss.total.reshape((plan.model_count,))),
        stopped(evaluation.feasibility.feasible.reshape((plan.model_count,))),
        stopped(indices),
        stopped(weights),
        stopped(support),
        realization_ids,
        posterior_id,
        campaign_id,
        batch_id,
        plan.plan_id,
        canonical_fingerprint(
            {"kind": "sampling-mpc-selected-replay", "plan": plan.plan_id}
        ),
    )


def _aggregate(
    plan: SamplingMPCPlan,
    objectives: Array,
    weights: Array,
    support: Array,
    /,
) -> Array:
    supported_maximum = jnp.max(jnp.where(support, objectives, -jnp.inf))
    safe_objectives = jnp.where(
        support,
        objectives,
        jax.lax.stop_gradient(supported_maximum),
    )
    if plan.aggregation == "expectation":
        return jnp.sum(weights * safe_objectives)
    if plan.aggregation == "worst_case":
        return supported_maximum
    assert plan.risk_measure is not None
    return plan.risk_measure.evaluate(safe_objectives, weights)


def _cem_update(
    plan: SamplingMPCPlan,
    mean: Array,
    deviation: Array,
    candidates: Array,
    valid: Array,
    elite_indices: Array,
    /,
) -> tuple[Array, Array]:
    if plan.update == "predictive":
        return mean, deviation
    elites = jax.lax.stop_gradient(candidates[elite_indices])
    elite_valid = valid[elite_indices]
    payload_axes = (plan.elite_count,) + (1,) * len(plan.parameter_shape)
    weights = elite_valid.astype(mean.dtype).reshape(payload_axes)
    count = jnp.sum(elite_valid.astype(mean.dtype))
    safe_count = jnp.maximum(count, 1.0)
    elite_mean = jnp.sum(jnp.where(weights > 0.0, elites, 0.0), axis=0) / safe_count
    centered = jnp.where(weights > 0.0, elites - elite_mean, 0.0)
    elite_variance = jnp.sum(jnp.square(centered), axis=0) / safe_count
    elite_deviation = jnp.maximum(
        jnp.sqrt(jnp.maximum(elite_variance, 0.0)),
        jnp.asarray(plan.minimum_standard_deviation, dtype=mean.dtype),
    )
    updated_mean = mean + plan.update_rate * (elite_mean - mean)
    updated_deviation = deviation + plan.update_rate * (elite_deviation - deviation)
    any_valid = count > 0.0
    return (
        jnp.where(any_valid, updated_mean, mean),
        jnp.where(any_valid, updated_deviation, deviation),
    )


def solve_sampling_mpc(
    plan: SamplingMPCPlan,
    state: SamplingMPCState,
    key: Array,
    /,
    *,
    warm_start: ArrayLike = False,
) -> SamplingMPCResult:
    """Run fixed Gaussian work and return the best valid sampled control.

    This routine reports only the best candidate observed in its finite work.
    It does not claim global optimality or certified robustness.
    """

    _validate_state(plan, state)
    key_ = jnp.asarray(key)
    typed_key = jax.dtypes.issubdtype(key_.dtype, jax.dtypes.prng_key)
    legacy_key = key_.dtype == jnp.dtype(jnp.uint32) and key_.shape == (2,)
    if typed_key and key_.shape != ():
        raise ValueError("A typed PRNG key must be scalar.")
    if not typed_key and not legacy_key:
        raise TypeError(
            "key must be a scalar typed PRNG key or a uint32 key of shape (2,)."
        )
    (
        realization_parameters,
        realization_weights,
        realization_support,
        realization_indices,
    ) = _realization_inputs(plan, key_, state.solve_count)
    shifted = shift_sampling_mpc_state(plan, state)
    use_shift = jnp.asarray(warm_start, dtype=bool).reshape(())
    initial_mean = jnp.where(use_shift, shifted.mean, state.mean)
    initial_deviation = jnp.where(
        use_shift, shifted.standard_deviation, state.standard_deviation
    )
    initial_controls = initial_mean
    initial_objective = jnp.asarray(jnp.inf, dtype=initial_mean.dtype)
    initial_valid = jnp.asarray(False)
    initial_index = jnp.asarray(-1, dtype=jnp.int32)

    def iteration(carry, index):
        (
            mean,
            deviation,
            best_controls,
            best_objective,
            best_valid,
            best_iteration,
            best_candidate,
        ) = carry
        noise = jax.random.normal(
            jax.random.fold_in(key_, index),
            (plan.candidate_count,) + plan.parameter_shape,
            dtype=mean.dtype,
        )
        noise = noise.at[0].set(jnp.zeros(plan.parameter_shape, dtype=mean.dtype))
        raw_candidates = mean[None] + deviation[None] * noise
        candidates, bounds_valid = _candidate_bounds(plan, raw_candidates)
        model_objectives, model_rollout_valid, model_feasible = jax.vmap(
            lambda candidate: _evaluate_candidate(
                plan,
                candidate,
                realization_parameters,
            )
        )(candidates)
        candidate_objectives = jax.vmap(
            lambda values: _aggregate(
                plan,
                values,
                realization_weights,
                realization_support,
            )
        )(model_objectives)
        candidate_accepted = (
            bounds_valid
            & jnp.all(
                (~realization_support)[None, :] | (model_rollout_valid & model_feasible),
                axis=-1,
            )
            & jnp.isfinite(candidate_objectives)
        )
        masked_objectives = jnp.where(candidate_accepted, candidate_objectives, jnp.inf)
        iteration_candidate = jnp.argmin(masked_objectives).astype(jnp.int32)
        iteration_objective = masked_objectives[iteration_candidate]
        iteration_valid = candidate_accepted[iteration_candidate]
        better = iteration_valid & (
            (~best_valid) | (iteration_objective < best_objective)
        )
        best_controls = jnp.where(better, candidates[iteration_candidate], best_controls)
        best_objective = jnp.where(better, iteration_objective, best_objective)
        best_valid = best_valid | iteration_valid
        best_iteration = jnp.where(better, index, best_iteration).astype(jnp.int32)
        best_candidate = jnp.where(better, iteration_candidate, best_candidate).astype(
            jnp.int32
        )

        _, elite_indices = jax.lax.top_k(-masked_objectives, plan.elite_count)
        elite_indices = jax.lax.stop_gradient(elite_indices)
        elite_accepted = candidate_accepted[elite_indices]
        next_mean, next_deviation = _cem_update(
            plan,
            mean,
            deviation,
            candidates,
            candidate_accepted,
            elite_indices,
        )
        next_carry = (
            next_mean,
            next_deviation,
            best_controls,
            best_objective,
            best_valid,
            best_iteration,
            best_candidate,
        )
        history = (
            candidates,
            model_objectives,
            model_rollout_valid,
            model_feasible,
            candidate_objectives,
            candidate_accepted,
            elite_indices,
            elite_accepted,
            next_mean,
            next_deviation,
        )
        return next_carry, history

    initial_carry = (
        initial_mean,
        initial_deviation,
        initial_controls,
        initial_objective,
        initial_valid,
        initial_index,
        initial_index,
    )
    final, history = jax.lax.scan(
        iteration,
        initial_carry,
        jnp.arange(plan.iteration_count, dtype=jnp.int32),
    )
    (
        final_mean,
        final_deviation,
        selected_controls,
        selected_objective,
        selected_valid,
        selected_iteration,
        selected_candidate,
    ) = final
    final_mean = jax.lax.stop_gradient(final_mean)
    final_deviation = jax.lax.stop_gradient(final_deviation)
    selected_controls = jax.lax.stop_gradient(selected_controls)
    selected_objective = jax.lax.stop_gradient(selected_objective)
    selected_iteration = jax.lax.stop_gradient(selected_iteration)
    selected_candidate = jax.lax.stop_gradient(selected_candidate)
    (
        candidate_history,
        model_objective_history,
        model_rollout_valid_history,
        model_feasible_history,
        objective_history,
        accepted_history,
        elite_index_history,
        elite_accepted_history,
        mean_tail,
        deviation_tail,
    ) = history
    mean_history = jnp.concatenate((initial_mean[None], mean_tail), axis=0)
    deviation_history = jnp.concatenate((initial_deviation[None], deviation_tail), axis=0)
    completed_iterations = jnp.asarray(plan.iteration_count, dtype=jnp.int32)
    candidate_evaluations = jnp.asarray(
        plan.iteration_count * plan.candidate_count, dtype=jnp.int32
    )
    model_rollouts = jnp.asarray(
        plan.iteration_count * plan.candidate_count * plan.model_count + plan.model_count,
        dtype=jnp.int32,
    )
    (
        realization_ids,
        posterior_id,
        campaign_id,
        realization_policy,
        realization_batch_id,
    ) = _realization_metadata(plan)
    evidence = SamplingMPCEvidence(
        candidate_history,
        model_objective_history,
        model_rollout_valid_history,
        model_feasible_history,
        objective_history,
        accepted_history,
        elite_index_history,
        elite_accepted_history,
        mean_history,
        deviation_history,
        jnp.broadcast_to(
            realization_indices,
            (plan.iteration_count, plan.model_count),
        ),
        realization_weights,
        realization_support,
        completed_iterations,
        candidate_evaluations,
        model_rollouts,
        1,
        2,
        plan.model_shape,
        realization_ids,
        posterior_id,
        campaign_id,
        realization_policy,
        realization_batch_id,
        plan.realization_binding_id,
        plan.plan_id,
        canonical_fingerprint({"kind": "sampling-mpc-evidence", "plan": plan.plan_id}),
    )
    final_state = SamplingMPCState(
        final_mean,
        final_deviation,
        state.solve_count + jnp.asarray(1, dtype=jnp.int32),
        state.plan_id,
        state.state_id,
    )
    replay = _selected_replay(
        plan,
        selected_controls,
        realization_parameters,
        realization_weights,
        realization_support,
        realization_indices,
    )
    action = jax.lax.stop_gradient(
        plan.parameterization.evaluate(
            selected_controls,
            plan.problem.time_grid.t0,
            case_shape=(),
        )
    )
    status = jnp.where(
        selected_valid,
        int(SamplingMPCStatus.SUCCESS),
        int(SamplingMPCStatus.NO_VALID_CANDIDATE),
    ).astype(jnp.int32)
    return SamplingMPCResult(
        final_state,
        evidence,
        replay,
        selected_controls,
        action,
        selected_objective,
        selected_valid,
        status,
        selected_iteration,
        selected_candidate,
        jnp.asarray(True),
        plan.problem_id,
        plan.parameterization_id,
        plan.plan_id,
        canonical_fingerprint({"kind": "sampling-mpc-result", "plan": plan.plan_id}),
        "control:sampling-mpc:predictive-gaussian",
    )


__all__ = [
    "SamplingMPCAggregation",
    "SamplingMPCBoundPolicy",
    "SamplingMPCEvidence",
    "SamplingMPCPlan",
    "SamplingMPCRealizationBinding",
    "SamplingMPCRealizationPolicy",
    "SamplingMPCRealizations",
    "SamplingMPCResult",
    "SamplingMPCSelectedReplay",
    "SamplingMPCState",
    "SamplingMPCStatus",
    "SamplingMPCUpdate",
    "SamplingMPCWarmStartTerminal",
    "initialize_sampling_mpc",
    "plan_sampling_mpc",
    "shift_sampling_mpc_state",
    "solve_sampling_mpc",
]
