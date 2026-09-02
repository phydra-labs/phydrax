#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import array_tree_signature, canonical_fingerprint
from .._numerics._checkpointed_scan import (
    AdaptiveReplayPreparationPolicy,
    checkpointed_scan,
    prepare_replay_schedule,
    PreparedReplaySchedule,
)
from .._numerics._ssp_runge_kutta import (
    AbstractSSPRKStageTransform,
    ssprk33_step_with_evidence,
    ssprk54_step_with_evidence,
    SSPRKStepResult,
    StageTransformResult,
)
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ..discretization import DiscretizationBundle
from ..metrix import AbstractStateGeometry, EuclideanStateGeometry


def _canonical_structured_state(state: Any, /) -> PyTree[Array]:
    leaves, treedef = jax.tree.flatten(state)
    if not leaves:
        raise ValueError("Fixed-step initial_state must contain array leaves.")
    if any(not eqx.is_array(leaf) for leaf in leaves):
        raise TypeError("Every structured fixed-step state leaf must be an array.")
    arrays = tuple(jnp.asarray(leaf) for leaf in leaves)
    if not any(jnp.issubdtype(array.dtype, jnp.inexact) for array in arrays):
        raise TypeError(
            "Structured fixed-step initial_state requires at least one inexact leaf."
        )
    return jax.tree.unflatten(treedef, arrays)


def _state_dtype(state: PyTree[Array], /):
    dtypes = tuple(
        leaf.dtype
        for leaf in jax.tree.leaves(state)
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
    )
    if not dtypes:
        raise TypeError("Fixed-step state requires at least one inexact leaf.")
    return jnp.result_type(*dtypes)


def _validate_result_state(
    role: str, candidate: PyTree[Array], reference: PyTree[Array], /
) -> None:
    if jax.tree.structure(candidate) != jax.tree.structure(reference):
        raise ValueError(f"Fixed-step {role} must preserve the state PyTree structure.")
    for proposed, current in zip(
        jax.tree.leaves(candidate), jax.tree.leaves(reference), strict=True
    ):
        if not eqx.is_array(proposed):
            raise TypeError(f"Every fixed-step {role} leaf must be an array.")
        if proposed.shape != current.shape or proposed.dtype != current.dtype:
            raise ValueError(
                f"Fixed-step {role} must preserve every state leaf shape and dtype."
            )


def _validate_scalar_result(role: str, value: Any, /, *, boolean: bool = False) -> None:
    if not eqx.is_array(value) or value.shape != ():
        raise TypeError(f"Fixed-step {role} must be a scalar array.")
    if boolean and value.dtype != jnp.dtype(bool):
        raise TypeError(f"Fixed-step {role} must be Boolean.")


def _prepend_initial_state(
    initial: PyTree[Array], states: PyTree[Array], /
) -> PyTree[Array]:
    return jax.tree.map(
        lambda first, rest: jnp.concatenate((first[None, ...], rest), axis=0),
        initial,
        states,
    )


def _take_saved_states(states: PyTree[Array], indices: Array, /) -> PyTree[Array]:
    return jax.tree.map(lambda leaf: leaf[indices], states)


class AcceptedStepTransformResult(StrictModule):
    transformed_state: Array
    applied: Array
    successful: Array
    correction_norm: Array


class AbstractAcceptedStepTransform(StrictModule, NonTrainableState):
    transform_id: AbstractAttribute[str]

    @abc.abstractmethod
    def apply(
        self,
        step_index: Array,
        time: Array,
        previous_state: Array,
        candidate_state: Array,
        args: Any,
        /,
    ) -> AcceptedStepTransformResult:
        raise NotImplementedError


class IdentityAcceptedStepTransform(AbstractAcceptedStepTransform):
    transform_id: str = "accepted-step-transform:identity"

    def apply(
        self,
        step_index: Array,
        time: Array,
        previous_state: Array,
        candidate_state: Array,
        args: Any,
        /,
    ) -> AcceptedStepTransformResult:
        del step_index, time, previous_state, args
        return AcceptedStepTransformResult(
            candidate_state,
            jnp.asarray(False),
            jnp.asarray(True),
            jnp.zeros((), dtype=candidate_state.dtype),
        )


class CompositeAcceptedStepTransform(AbstractAcceptedStepTransform):
    transforms: tuple[AbstractAcceptedStepTransform, ...]
    transform_id: str = eqx.field(static=True)

    def __init__(self, transforms: Sequence[AbstractAcceptedStepTransform], /):
        values = tuple(transforms)
        if any(not isinstance(value, AbstractAcceptedStepTransform) for value in values):
            raise TypeError("Every transform must be an AbstractAcceptedStepTransform.")
        self.transforms = values
        self.transform_id = canonical_fingerprint(
            {
                "kind": "composite-accepted-step-transform",
                "transforms": [value.transform_id for value in values],
            }
        )

    def apply(
        self,
        step_index: Array,
        time: Array,
        previous_state: Array,
        candidate_state: Array,
        args: Any,
        /,
    ) -> AcceptedStepTransformResult:
        state = candidate_state
        applied = jnp.asarray(False)
        successful = jnp.asarray(True)
        correction = jnp.zeros((), dtype=candidate_state.dtype)
        for transform in self.transforms:
            result = transform.apply(step_index, time, previous_state, state, args)
            state = jnp.where(result.successful, result.transformed_state, state)
            applied = applied | result.applied
            successful = successful & result.successful
            correction = correction + result.correction_norm
        return AcceptedStepTransformResult(state, applied, successful, correction)


class IdentitySSPRKStageTransform(AbstractSSPRKStageTransform):
    transform_id: str = "ssprk-stage-transform:identity"

    def apply(
        self,
        stage_index: int,
        time: Array,
        candidate_state: Array,
        args: Any,
        /,
    ) -> StageTransformResult:
        del stage_index, time, args
        return StageTransformResult(
            candidate_state,
            jnp.asarray(False),
            jnp.asarray(True),
            jnp.zeros((), dtype=candidate_state.real.dtype),
        )


class CallableSSPRKStageTransform(AbstractSSPRKStageTransform):
    transform: Callable[[int, Array, Array, Any], StageTransformResult] = eqx.field(
        static=True
    )
    transform_id: str = eqx.field(static=True)

    def __init__(self, transform, transform_id: str, /):
        if not callable(transform):
            raise TypeError("transform must be callable.")
        identifier = str(transform_id)
        if not identifier:
            raise ValueError("transform_id must be non-empty.")
        self.transform = transform
        self.transform_id = identifier

    def apply(
        self,
        stage_index: int,
        time: Array,
        candidate_state: Array,
        args: Any,
        /,
    ) -> StageTransformResult:
        result = self.transform(stage_index, time, candidate_state, args)
        if not isinstance(result, StageTransformResult):
            raise TypeError("Callable stage transforms must return StageTransformResult.")
        return result


class FixedStepResult(StrictModule):
    candidate_state: PyTree[Array]
    accepted_state: PyTree[Array]
    successful: Array
    residual: Array
    iterations: Array
    work: Array
    transform_applied: Array
    transform_correction_norm: Array


class RobustRetryPolicy(StrictModule, NonTrainableState):
    maximum_retries: int = eqx.field(static=True)
    reduction_factor: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_retries: int = 4,
        reduction_factor: float = 0.5,
    ):
        retries = int(maximum_retries)
        factor = float(reduction_factor)
        if retries < 0 or not 0.0 < factor < 1.0:
            raise ValueError("Retry policy requires retries>=0 and reduction in (0,1).")
        self.maximum_retries = retries
        self.reduction_factor = factor
        self.policy_id = canonical_fingerprint(
            {
                "kind": "robust-fixed-step-retry-policy",
                "maximum_retries": retries,
                "reduction_factor": factor,
                "differentiability": "branchwise",
            }
        )


class RetriedFixedStepResult(StrictModule):
    candidate_state: Array
    accepted_state: Array
    successful: Array
    accepted_step_size: Array
    retry_count: Array
    attempted_step_sizes: Array
    decision_id: str = eqx.field(static=True)


class AbstractFixedStepMethod(StrictModule, NonTrainableState):
    method_id: AbstractAttribute[str]

    @abc.abstractmethod
    def step(
        self,
        step_index: Array,
        time: Array,
        state: PyTree[Array],
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        raise NotImplementedError


class CallableFixedStepMethod(AbstractFixedStepMethod):
    step_function: Callable[[Array, Array, PyTree[Array], Array, Any], FixedStepResult]
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        step_function: Callable[
            [Array, Array, PyTree[Array], Array, Any], FixedStepResult
        ],
        method_id: str,
        /,
    ):
        if not callable(step_function):
            raise TypeError("step_function must be callable.")
        identifier = str(method_id)
        if not identifier:
            raise ValueError("method_id must be non-empty.")
        self.step_function = step_function
        self.method_id = identifier

    def step(
        self,
        step_index: Array,
        time: Array,
        state: PyTree[Array],
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        result = self.step_function(step_index, time, state, step_size, args)
        if not isinstance(result, FixedStepResult):
            raise TypeError("step_function must return FixedStepResult.")
        return result


class AbstractSSPRKFixedStepMethod(AbstractFixedStepMethod):
    vector_field: Callable[[Array, Array, Any], Array]
    transform: AbstractAcceptedStepTransform
    stage_transform: AbstractSSPRKStageTransform
    order: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        vector_field: Callable[[Array, Array, Any], Array],
        /,
        *,
        order: int,
        transform: AbstractAcceptedStepTransform | None = None,
        stage_transform: AbstractSSPRKStageTransform | None = None,
    ):
        if not callable(vector_field):
            raise TypeError("vector_field must be callable.")
        if order not in (3, 4):
            raise ValueError("Fixed-step SSPRK order must be 3 or 4.")
        transform_ = IdentityAcceptedStepTransform() if transform is None else transform
        if not isinstance(transform_, AbstractAcceptedStepTransform):
            raise TypeError("transform must be an AbstractAcceptedStepTransform or None.")
        stage_transform_ = (
            IdentitySSPRKStageTransform() if stage_transform is None else stage_transform
        )
        if not isinstance(stage_transform_, AbstractSSPRKStageTransform):
            raise TypeError(
                "stage_transform must be AbstractSSPRKStageTransform or None."
            )
        self.vector_field = vector_field
        self.transform = transform_
        self.stage_transform = stage_transform_
        self.order = int(order)
        self.method_id = canonical_fingerprint(
            {
                "kind": "fixed-step-ssprk",
                "order": order,
                "transform": transform_.transform_id,
                "stage_transform": stage_transform_.transform_id,
            }
        )

    @abc.abstractmethod
    def _advance(
        self,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> SSPRKStepResult:
        raise NotImplementedError

    def step(
        self,
        step_index: Array,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        advanced = self._advance(time, state, step_size, args)
        candidate = advanced.state
        transformed = self.transform.apply(
            step_index, time + step_size, state, candidate, args
        )
        successful = (
            advanced.successful
            & transformed.successful
            & jnp.all(jnp.isfinite(transformed.transformed_state))
        )
        accepted = jnp.where(successful, transformed.transformed_state, state)
        return FixedStepResult(
            candidate,
            accepted,
            successful,
            jnp.zeros((), dtype=state.dtype),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(self.order, dtype=jnp.int32),
            advanced.applied | transformed.applied,
            jnp.maximum(advanced.correction_norm, transformed.correction_norm),
        )


class SSPRK33FixedStepMethod(AbstractSSPRKFixedStepMethod):
    def __init__(
        self,
        vector_field: Callable[[Array, Array, Any], Array],
        /,
        *,
        transform: AbstractAcceptedStepTransform | None = None,
        stage_transform: AbstractSSPRKStageTransform | None = None,
    ):
        super().__init__(
            vector_field,
            order=3,
            transform=transform,
            stage_transform=stage_transform,
        )

    def _advance(self, time, state, step_size, args, /):
        return ssprk33_step_with_evidence(
            self.vector_field,
            time,
            state,
            step_size,
            args,
            stage_transform=self.stage_transform,
        )


class SSPRK54FixedStepMethod(AbstractSSPRKFixedStepMethod):
    def __init__(
        self,
        vector_field: Callable[[Array, Array, Any], Array],
        /,
        *,
        transform: AbstractAcceptedStepTransform | None = None,
        stage_transform: AbstractSSPRKStageTransform | None = None,
    ):
        super().__init__(
            vector_field,
            order=4,
            transform=transform,
            stage_transform=stage_transform,
        )

    def _advance(self, time, state, step_size, args, /):
        return ssprk54_step_with_evidence(
            self.vector_field,
            time,
            state,
            step_size,
            args,
            stage_transform=self.stage_transform,
        )


class FixedStepProblem(StrictModule, NonTrainableState):
    method: AbstractFixedStepMethod
    initial_state: PyTree[Array]
    args: Any
    state_geometry: AbstractStateGeometry
    discretization_bundle: DiscretizationBundle | None
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractFixedStepMethod,
        initial_state: Any,
        /,
        *,
        t0: float,
        t1: float,
        step_size: float,
        args: Any = None,
        state_geometry: AbstractStateGeometry | None = None,
        discretization_bundle: DiscretizationBundle | None = None,
        problem_id: str | None = None,
    ):
        if not isinstance(method, AbstractFixedStepMethod):
            raise TypeError("method must be an AbstractFixedStepMethod.")
        if state_geometry is None:
            initial = jnp.asarray(initial_state)
            if not jnp.issubdtype(initial.dtype, jnp.inexact):
                raise TypeError("Fixed-step initial_state must have an inexact dtype.")
        else:
            initial = _canonical_structured_state(initial_state)
        start = float(t0)
        end = float(t1)
        step = float(step_size)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("Fixed-step times require finite t1 > t0.")
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        raw_steps = (end - start) / step
        count = int(round(raw_steps))
        if count <= 0 or not np.isclose(raw_steps, count, rtol=1e-12, atol=1e-12):
            raise ValueError("Fixed-step interval must contain an integer step count.")
        geometry = EuclideanStateGeometry() if state_geometry is None else state_geometry
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("state_geometry must be an AbstractStateGeometry or None.")
        if discretization_bundle is not None and not isinstance(
            discretization_bundle, DiscretizationBundle
        ):
            raise TypeError(
                "discretization_bundle must be a DiscretizationBundle or None."
            )
        state_payload = (
            {
                "state_shape": list(initial.shape),
                "state_dtype": str(initial.dtype),
            }
            if eqx.is_array(initial)
            else {"state_tree": array_tree_signature(initial)}
        )
        generated = canonical_fingerprint(
            {
                "kind": "fixed-step-problem",
                "method": method.method_id,
                **state_payload,
                "t0": start,
                "t1": end,
                "step_size": step,
                "geometry": geometry.geometry_id,
                "bundle": None
                if discretization_bundle is None
                else discretization_bundle.bundle_id,
            }
        )
        identifier = generated if problem_id is None else str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.method = method
        self.initial_state = initial
        self.args = args
        self.state_geometry = geometry
        self.discretization_bundle = discretization_bundle
        self.t0 = start
        self.t1 = end
        self.step_size = step
        self.step_count = count
        self.problem_id = identifier


def retry_fixed_step(
    method: AbstractFixedStepMethod,
    policy: RobustRetryPolicy,
    step_index: Array,
    time: Array,
    state: Array,
    step_size: Array,
    args: Any = None,
    /,
) -> RetriedFixedStepResult:
    if not isinstance(method, AbstractFixedStepMethod) or not isinstance(
        policy, RobustRetryPolicy
    ):
        raise TypeError("retry_fixed_step requires method and retry policy.")
    initial = jnp.asarray(state)
    current_step = jnp.asarray(step_size)
    successful = jnp.asarray(False)
    selected_state = initial
    selected_candidate = initial
    accepted_step = jnp.zeros_like(current_step)
    retry_count = jnp.asarray(policy.maximum_retries, dtype=jnp.int32)
    attempted = []
    for attempt in range(policy.maximum_retries + 1):
        attempted.append(current_step)
        result = method.step(
            step_index,
            time,
            initial,
            current_step,
            args,
        )
        take = (~jax.lax.stop_gradient(successful)) & jax.lax.stop_gradient(
            result.successful
        )
        selected_candidate = jnp.where(take, result.candidate_state, selected_candidate)
        selected_state = jnp.where(take, result.accepted_state, selected_state)
        accepted_step = jnp.where(take, current_step, accepted_step)
        retry_count = jnp.where(take, jnp.asarray(attempt, dtype=jnp.int32), retry_count)
        successful = successful | result.successful
        current_step = current_step * policy.reduction_factor
    return RetriedFixedStepResult(
        selected_candidate,
        jnp.where(successful, selected_state, initial),
        successful,
        accepted_step,
        retry_count,
        jnp.stack(tuple(attempted)),
        canonical_fingerprint(
            {
                "kind": "retried-fixed-step-decision",
                "method": method.method_id,
                "retry_policy": policy.policy_id,
            }
        ),
    )


class FixedStepSolution(StrictModule, NonTrainableState):
    times: Array
    states: PyTree[Array]
    valid: Array
    successful: Array
    residuals: Array
    iterations: Array
    work: Array
    transform_applied: Array
    transform_correction_norm: Array
    problem_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    state_geometry_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)


FixedStepRetentionPolicy: TypeAlias = Literal["final", "checkpoints", "trajectory"]
FixedStepReplayMode: TypeAlias = Literal["full", "step", "block", "scheduled"]


class FixedStepReplayPolicy(StrictModule, NonTrainableState):
    """Reverse-mode storage and immutable recomputation for fixed-step scans."""

    mode: FixedStepReplayMode = eqx.field(static=True)
    block_size: int | None = eqx.field(static=True)
    schedule: PreparedReplaySchedule | None
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: FixedStepReplayMode = "full",
        /,
        *,
        block_size: int | None = None,
        schedule: PreparedReplaySchedule | None = None,
    ):
        if mode not in ("full", "step", "block", "scheduled"):
            raise ValueError("Unknown fixed-step replay mode.")
        size = None if block_size is None else int(block_size)
        if mode == "block":
            if size is None or size <= 0:
                raise ValueError("Block replay requires a positive block_size.")
            if schedule is not None:
                raise ValueError("Block replay does not accept a prepared schedule.")
        elif mode == "scheduled":
            if size is not None:
                raise ValueError("Scheduled replay does not accept block_size.")
            if not isinstance(schedule, PreparedReplaySchedule):
                raise TypeError("Scheduled replay requires PreparedReplaySchedule.")
        elif size is not None or schedule is not None:
            raise ValueError("Replay block/schedule is incompatible with selected mode.")
        self.mode = mode
        self.block_size = size
        self.schedule = schedule
        self.policy_id = canonical_fingerprint(
            {
                "kind": "fixed-step-replay",
                "mode": mode,
                "block_size": size,
                "schedule": None if schedule is None else schedule.schedule_id,
            }
        )


FixedStepScalarDiagnostics: TypeAlias = Callable[
    [Array, Array, PyTree[Array], Any], PyTree[Array]
]


def _fixed_step_advance(
    problem: FixedStepProblem,
    state_dtype: Any,
    carry: tuple[PyTree[Array], Array],
    step_index: Array,
    /,
):
    state, previous_success = carry
    step_size = jnp.asarray(problem.step_size, dtype=state_dtype)
    time = jnp.asarray(problem.t0, dtype=state_dtype) + step_index * step_size
    result = problem.method.step(step_index, time, state, step_size, problem.args)
    if not isinstance(result, FixedStepResult):
        raise TypeError("Fixed-step methods must return FixedStepResult.")
    _validate_result_state("candidate_state", result.candidate_state, state)
    _validate_result_state("accepted_state", result.accepted_state, state)
    _validate_scalar_result("successful", result.successful, boolean=True)
    _validate_scalar_result("residual", result.residual)
    _validate_scalar_result("iterations", result.iterations)
    _validate_scalar_result("work", result.work)
    _validate_scalar_result("transform_applied", result.transform_applied, boolean=True)
    _validate_scalar_result("transform_correction_norm", result.transform_correction_norm)
    accepted = tree_where(previous_success, result.accepted_state, state)
    successful = previous_success & result.successful
    payload = (
        successful,
        result.residual,
        result.iterations,
        result.work,
        result.transform_applied,
        result.transform_correction_norm,
    )
    return (accepted, successful), payload


def _validate_scalar_diagnostics(diagnostics: PyTree[Array], /) -> None:
    leaves = jax.tree.leaves(diagnostics)
    if not leaves:
        raise ValueError("Fixed-step diagnostics must contain scalar array leaves.")
    if any(not eqx.is_array(leaf) or leaf.shape != () for leaf in leaves):
        raise TypeError("Every fixed-step diagnostic leaf must be a scalar array.")


class FixedStepRolloutResult(StrictModule, NonTrainableState):
    final_state: PyTree[Array]
    successful: Array
    times: Array
    states: PyTree[Array]
    valid: Array
    residuals: Array
    iterations: Array
    work: Array
    transform_applied: Array
    transform_correction_norm: Array
    diagnostics: Any
    problem_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    state_geometry_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class FixedStepRolloutPlan(StrictModule, NonTrainableState):
    """Fixed-step retention with an orthogonal deterministic replay policy."""

    retention: FixedStepRetentionPolicy = eqx.field(static=True)
    checkpoint_stride: int = eqx.field(static=True)
    replay: FixedStepReplayPolicy
    diagnostics: FixedStepScalarDiagnostics | None
    diagnostics_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        retention: FixedStepRetentionPolicy = "final",
        checkpoint_stride: int = 1,
        replay: FixedStepReplayPolicy | None = None,
        diagnostics: FixedStepScalarDiagnostics | None = None,
        diagnostics_id: str | None = None,
    ):
        if retention not in ("final", "checkpoints", "trajectory"):
            raise ValueError("Unknown fixed-step retention policy.")
        stride = int(checkpoint_stride)
        if stride <= 0:
            raise ValueError("checkpoint_stride must be positive.")
        if retention != "checkpoints" and stride != 1:
            raise ValueError(
                "checkpoint_stride differs from one only for checkpoint retention."
            )
        replay_ = FixedStepReplayPolicy() if replay is None else replay
        if not isinstance(replay_, FixedStepReplayPolicy):
            raise TypeError("replay must be FixedStepReplayPolicy or None.")
        if diagnostics is not None and not callable(diagnostics):
            raise TypeError("diagnostics must be callable or None.")
        if diagnostics is None:
            if diagnostics_id is not None:
                raise ValueError("diagnostics_id requires a diagnostics callback.")
            diagnostic_identifier = None
        else:
            diagnostic_identifier = "" if diagnostics_id is None else str(diagnostics_id)
            if not diagnostic_identifier:
                raise ValueError(
                    "A diagnostics callback requires a non-empty diagnostics_id."
                )
        self.retention = retention
        self.checkpoint_stride = stride
        self.replay = replay_
        self.diagnostics = diagnostics
        self.diagnostics_id = diagnostic_identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-step-rollout-plan",
                "retention": retention,
                "checkpoint_stride": stride,
                "replay": replay_.policy_id,
                "diagnostics": diagnostic_identifier,
            }
        )

    def rollout(
        self,
        problem: FixedStepProblem,
        /,
    ) -> FixedStepRolloutResult:
        if not isinstance(problem, FixedStepProblem):
            raise TypeError("problem must be a FixedStepProblem.")
        state_dtype = _state_dtype(problem.initial_state)
        step_size = jnp.asarray(problem.step_size, dtype=state_dtype)

        def advance(carry, step_index):
            next_carry, built_in = _fixed_step_advance(
                problem, state_dtype, carry, step_index
            )
            accepted, _ = next_carry
            if self.diagnostics is None:
                observed = ()
            else:
                endpoint = (
                    jnp.asarray(problem.t0, dtype=state_dtype)
                    + (step_index + 1) * step_size
                )
                observed = self.diagnostics(step_index, endpoint, accepted, problem.args)
                _validate_scalar_diagnostics(observed)
            return next_carry, (*built_in, observed)

        step = advance
        indices = jnp.arange(problem.step_count, dtype=jnp.int32)
        initial_carry = (problem.initial_state, jnp.asarray(True))

        if self.retention == "trajectory":

            def trajectory_step(carry, step_index):
                next_carry, payload = step(carry, step_index)
                return next_carry, (next_carry[0], *payload)

            (final_state, final_success), payload = checkpointed_scan(
                trajectory_step,
                initial_carry,
                indices,
                length=problem.step_count,
                mode=self.replay.mode,
                block_size=self.replay.block_size,
                schedule=self.replay.schedule,
            )
            (
                states,
                valid,
                residuals,
                iterations,
                work,
                transformed,
                correction,
                observed,
            ) = payload
            retained_states = _prepend_initial_state(problem.initial_state, states)
            retained_valid = jnp.concatenate((jnp.asarray([True]), valid), axis=0)
            retained_times = jnp.asarray(
                problem.t0, dtype=step_size.dtype
            ) + step_size * jnp.arange(problem.step_count + 1)
        elif self.retention == "final":
            (final_state, final_success), payload = checkpointed_scan(
                step,
                initial_carry,
                indices,
                length=problem.step_count,
                mode=self.replay.mode,
                block_size=self.replay.block_size,
                schedule=self.replay.schedule,
            )
            valid, residuals, iterations, work, transformed, correction, observed = (
                payload
            )
            retained_states = jax.tree.map(lambda leaf: leaf[None, ...], final_state)
            retained_valid = final_success[None]
            retained_times = jnp.asarray([problem.t1], dtype=step_size.dtype)
        else:
            saved_indices = tuple(
                range(0, problem.step_count + 1, self.checkpoint_stride)
            )
            if saved_indices[-1] != problem.step_count:
                saved_indices = (*saved_indices, problem.step_count)
            save_after_step = np.zeros((problem.step_count,), dtype=bool)
            for endpoint in saved_indices[1:]:
                save_after_step[endpoint - 1] = True
            save_mask = jnp.asarray(save_after_step)
            retained_states = jax.tree.map(
                lambda leaf: (
                    jnp.zeros((len(saved_indices), *leaf.shape), dtype=leaf.dtype)
                    .at[0]
                    .set(leaf)
                ),
                problem.initial_state,
            )
            retained_valid = jnp.zeros((len(saved_indices),), dtype=bool).at[0].set(True)

            def checkpoint_step(carry, step_index):
                state_carry, saved, saved_valid, cursor = carry
                next_carry, payload = step(state_carry, step_index)
                accepted, successful = next_carry

                def store(values):
                    states_, valid_, cursor_ = values
                    states_ = jax.tree.map(
                        lambda buffer, value: buffer.at[cursor_].set(value),
                        states_,
                        accepted,
                    )
                    valid_ = valid_.at[cursor_].set(successful)
                    return states_, valid_, cursor_ + 1

                saved, saved_valid, cursor = jax.lax.cond(
                    save_mask[step_index],
                    store,
                    lambda values: values,
                    (saved, saved_valid, cursor),
                )
                return (next_carry, saved, saved_valid, cursor), payload

            checkpoint_carry = (
                initial_carry,
                retained_states,
                retained_valid,
                jnp.asarray(1, dtype=jnp.int32),
            )
            result_carry, payload = checkpointed_scan(
                checkpoint_step,
                checkpoint_carry,
                indices,
                length=problem.step_count,
                mode=self.replay.mode,
                block_size=self.replay.block_size,
                schedule=self.replay.schedule,
            )
            (final_state, final_success), retained_states, retained_valid, _ = (
                result_carry
            )
            valid, residuals, iterations, work, transformed, correction, observed = (
                payload
            )
            retained_times = jnp.asarray(
                problem.t0, dtype=step_size.dtype
            ) + step_size * jnp.asarray(saved_indices, dtype=step_size.dtype)

        bundle_id = (
            None
            if problem.discretization_bundle is None
            else problem.discretization_bundle.bundle_id
        )
        return FixedStepRolloutResult(
            final_state,
            final_success,
            retained_times,
            retained_states,
            retained_valid,
            residuals,
            iterations,
            work,
            transformed,
            correction,
            observed,
            problem.problem_id,
            problem.method.method_id,
            problem.state_geometry.geometry_id,
            bundle_id,
            self.plan_id,
        )


def solve_fixed_step(
    problem: FixedStepProblem,
    /,
    *,
    save_every: int = 1,
    replay: FixedStepReplayPolicy | None = None,
) -> FixedStepSolution:
    """Run one pure fixed-step scan with orthogonal saving and replay policies."""

    if not isinstance(problem, FixedStepProblem):
        raise TypeError("problem must be a FixedStepProblem.")
    stride = int(save_every)
    if stride <= 0:
        raise ValueError("save_every must be positive.")
    replay_ = FixedStepReplayPolicy() if replay is None else replay
    if not isinstance(replay_, FixedStepReplayPolicy):
        raise TypeError("replay must be FixedStepReplayPolicy or None.")
    state_dtype = _state_dtype(problem.initial_state)
    step_size = jnp.asarray(problem.step_size, dtype=state_dtype)

    def advance(carry, step_index):
        next_carry, payload = _fixed_step_advance(problem, state_dtype, carry, step_index)
        return next_carry, (next_carry[0], *payload)

    indices = jnp.arange(problem.step_count, dtype=jnp.int32)
    (_, final_success), payload = checkpointed_scan(
        advance,
        (problem.initial_state, jnp.asarray(True)),
        indices,
        length=problem.step_count,
        mode=replay_.mode,
        block_size=replay_.block_size,
        schedule=replay_.schedule,
    )
    states, valid, residuals, iterations, work, transformed, correction = payload
    all_states = _prepend_initial_state(problem.initial_state, states)
    all_valid = jnp.concatenate((jnp.asarray([True]), valid), axis=0)
    all_times = jnp.asarray(problem.t0, dtype=step_size.dtype) + step_size * jnp.arange(
        problem.step_count + 1
    )
    save_indices = jnp.arange(0, problem.step_count + 1, stride, dtype=jnp.int32)
    if int(save_indices[-1]) != problem.step_count:
        save_indices = jnp.concatenate(
            (save_indices, jnp.asarray([problem.step_count], dtype=jnp.int32))
        )
    bundle_id = (
        None
        if problem.discretization_bundle is None
        else problem.discretization_bundle.bundle_id
    )
    return FixedStepSolution(
        all_times[save_indices],
        _take_saved_states(all_states, save_indices),
        all_valid[save_indices],
        final_success,
        residuals,
        iterations,
        work,
        transformed,
        correction,
        problem.problem_id,
        problem.method.method_id,
        problem.state_geometry.geometry_id,
        bundle_id,
    )


__all__ = [
    "AdaptiveReplayPreparationPolicy",
    "AbstractAcceptedStepTransform",
    "AbstractSSPRKStageTransform",
    "CallableFixedStepMethod",
    "CallableSSPRKStageTransform",
    "AbstractFixedStepMethod",
    "AcceptedStepTransformResult",
    "CompositeAcceptedStepTransform",
    "FixedStepProblem",
    "FixedStepReplayMode",
    "FixedStepReplayPolicy",
    "PreparedReplaySchedule",
    "prepare_replay_schedule",
    "FixedStepRetentionPolicy",
    "FixedStepRolloutPlan",
    "FixedStepRolloutResult",
    "FixedStepScalarDiagnostics",
    "FixedStepResult",
    "RetriedFixedStepResult",
    "RobustRetryPolicy",
    "retry_fixed_step",
    "FixedStepSolution",
    "IdentityAcceptedStepTransform",
    "IdentitySSPRKStageTransform",
    "SSPRK33FixedStepMethod",
    "SSPRK54FixedStepMethod",
    "StageTransformResult",
    "solve_fixed_step",
]
