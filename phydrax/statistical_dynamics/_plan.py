#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..continuation._core import ParameterContinuationProblem
from ._cumulants import (
    CumulantState,
    CumulantStateEvidence,
    DenseCumulantState,
    densify_cumulant,
    FactorCumulantState,
    factorize_cumulant,
    ForcingCovariance,
    RankAdaptationEvent,
    RankAdaptationPolicy,
    require_valid_state,
    SecondCumulantLayout,
    state_evidence,
)
from ._interactions import InteractionContinuationSchedule


ClosureKind: TypeAlias = Literal["ce2", "gce2"]


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


class QuadraticDynamics(StrictModule, NonTrainableState):
    """Finite dynamics ``c + A x + B[x,x]`` with an explicit tensor convention."""

    constant: Array
    linear: Array
    quadratic: Array
    dimension: int = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        constant: ArrayLike,
        linear: ArrayLike,
        quadratic: ArrayLike,
        /,
        *,
        dynamics_id: str | None = None,
    ):
        constant_ = jnp.asarray(constant)
        linear_ = jnp.asarray(linear)
        quadratic_ = jnp.asarray(quadratic)
        if constant_.ndim != 1 or constant_.shape[0] < 1:
            raise ValueError("Quadratic dynamics require a non-empty constant vector.")
        dimension = int(constant_.shape[0])
        if linear_.shape != (dimension, dimension) or quadratic_.shape != (
            dimension,
            dimension,
            dimension,
        ):
            raise ValueError("Linear and quadratic arrays have incompatible dimensions.")
        if not all(
            jnp.issubdtype(value.dtype, jnp.inexact)
            for value in (constant_, linear_, quadratic_)
        ):
            raise TypeError("Quadratic dynamics arrays must use inexact dtypes.")
        if not all(
            bool(np.asarray(jnp.all(jnp.isfinite(value))))
            for value in (constant_, linear_, quadratic_)
        ):
            raise ValueError("Quadratic dynamics arrays must be finite.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "quadratic-statistical-dynamics",
                    "constant": array_tree_fingerprint(constant_),
                    "linear": array_tree_fingerprint(linear_),
                    "quadratic": array_tree_fingerprint(quadratic_),
                    "tensor_convention": "B_i,j,k*x_j*x_k",
                }
            )
            if dynamics_id is None
            else str(dynamics_id)
        )
        if not identifier:
            raise ValueError("dynamics_id must be non-empty.")
        self.constant = constant_
        self.linear = linear_
        self.quadratic = quadratic_
        self.dimension = dimension
        self.dynamics_id = identifier

    def __call__(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != (self.dimension,):
            raise ValueError("Quadratic dynamics state has an incompatible shape.")
        return (
            self.constant
            + oe.contract("ij,j->i", self.linear, value)
            + oe.contract("ijk,j,k->i", self.quadratic, value, value)
        )

    def jacobian(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != (self.dimension,):
            raise ValueError("Quadratic dynamics state has an incompatible shape.")
        return (
            self.linear
            + oe.contract("ijk,k->ij", self.quadratic, value)
            + oe.contract("ikj,k->ij", self.quadratic, value)
        )

    def covariance_feedback(self, covariance: ArrayLike, /) -> Array:
        value = jnp.asarray(covariance)
        if value.shape != (self.dimension, self.dimension):
            raise ValueError("Embedded covariance has an incompatible shape.")
        return oe.contract("ijk,jk->i", self.quadratic, value)


class StatisticalDynamicsCost(StrictModule, NonTrainableState):
    state_dimension: int = eqx.field(static=True)
    mean_dimension: int = eqx.field(static=True)
    covariance_dimension: int = eqx.field(static=True)
    dense_state_bytes: int = eqx.field(static=True)
    dense_workspace_bytes: int = eqx.field(static=True)
    maximum_state_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)


class StatisticalDynamicsPlan(StrictModule, NonTrainableState):
    """Immutable exact CE2/GCE2 plan for declared quadratic dynamics.

    CE2 is admitted only for a QL partition and GCE2 only for a GQL partition.
    The eddy covariance equation is the exact ensemble second moment of that
    selected linear eddy equation; no third-cumulant term is silently dropped.
    """

    layout: SecondCumulantLayout
    dynamics: QuadraticDynamics
    forcing: ForcingCovariance
    closure: ClosureKind = eqx.field(static=True)
    interaction_model: str = eqx.field(static=True)
    closure_exact: bool = eqx.field(static=True)
    exactness: str = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    hermitian_tolerance: float = eqx.field(static=True)
    psd_tolerance: float = eqx.field(static=True)
    maximum_state_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: SecondCumulantLayout,
        dynamics: QuadraticDynamics,
        forcing: ForcingCovariance,
        /,
        *,
        closure: ClosureKind,
        interaction_model: Literal["ql", "gql"],
        time_step: float,
        hermitian_tolerance: float = 1.0e-10,
        psd_tolerance: float = 1.0e-10,
        maximum_state_bytes: int = 512 * 1024 * 1024,
        maximum_workspace_bytes: int = 2 * 1024 * 1024 * 1024,
    ):
        if not isinstance(layout, SecondCumulantLayout):
            raise TypeError("layout must be a SecondCumulantLayout.")
        if not isinstance(dynamics, QuadraticDynamics):
            raise TypeError("dynamics must be QuadraticDynamics.")
        if not isinstance(forcing, ForcingCovariance):
            raise TypeError("forcing must be ForcingCovariance.")
        if dynamics.dimension != layout.state_size:
            raise ValueError("Dynamics and second-cumulant layout dimensions differ.")
        if forcing.dimension != layout.eddy_dimension:
            raise ValueError("Forcing covariance must act on the eddy coordinates.")
        if (closure, interaction_model) not in (("ce2", "ql"), ("gce2", "gql")):
            raise ValueError(
                "CE2 requires QL and GCE2 requires GQL interaction selection."
            )
        step = float(time_step)
        hermitian = float(hermitian_tolerance)
        psd = float(psd_tolerance)
        state_limit = int(maximum_state_bytes)
        workspace_limit = int(maximum_workspace_bytes)
        if (
            not np.isfinite(step)
            or step <= 0.0
            or not np.isfinite(hermitian)
            or hermitian < 0.0
            or not np.isfinite(psd)
            or psd < 0.0
            or state_limit <= 0
            or workspace_limit <= 0
        ):
            raise ValueError(
                "Statistical-dynamics step, tolerances, and resources are invalid."
            )
        complex_quadratic = jnp.issubdtype(
            dynamics.quadratic.dtype, jnp.complexfloating
        ) and bool(
            np.max(np.abs(np.asarray(dynamics.quadratic)), initial=0.0) > hermitian
        )
        if complex_quadratic:
            raise ValueError(
                "Exact quadratic cumulants require independent real coordinates; "
                "convert Hermitian spectral states before preparing CE2/GCE2."
            )
        self.layout = layout
        self.dynamics = dynamics
        self.forcing = forcing
        self.closure = closure
        self.interaction_model = interaction_model
        self.closure_exact = True
        self.exactness = f"{closure}-exact-for-{interaction_model}"
        self.time_step = step
        self.hermitian_tolerance = hermitian
        self.psd_tolerance = psd
        self.maximum_state_bytes = state_limit
        self.maximum_workspace_bytes = workspace_limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "statistical-dynamics-plan",
                "layout": layout.layout_id,
                "dynamics": dynamics.dynamics_id,
                "forcing": forcing.covariance_id,
                "closure": closure,
                "interaction_model": interaction_model,
                "exactness": f"{closure}-exact-for-{interaction_model}",
                "time_step": step,
                "hermitian_tolerance": hermitian,
                "psd_tolerance": psd,
                "maximum_state_bytes": state_limit,
                "maximum_workspace_bytes": workspace_limit,
                "psd_repair": "none",
            }
        )

    def prepare(self, /) -> "PreparedStatisticalDynamics":
        dtype = np.dtype(
            jnp.result_type(
                self.dynamics.constant.dtype,
                self.dynamics.linear.dtype,
                self.dynamics.quadratic.dtype,
                self.forcing.covariance.dtype,
            )
        )
        mean_bytes = self.layout.mean_dimension * dtype.itemsize
        covariance_bytes = self.layout.eddy_dimension**2 * dtype.itemsize
        dense_state_bytes = mean_bytes + covariance_bytes
        dense_workspace_bytes = (
            8 * dense_state_bytes
            + (self.layout.eddy_dimension**2 + self.layout.state_size**2) * dtype.itemsize
        )
        if dense_state_bytes > self.maximum_state_bytes:
            raise MemoryError("Statistical-dynamics state exceeds maximum_state_bytes.")
        if dense_workspace_bytes > self.maximum_workspace_bytes:
            raise MemoryError(
                "Statistical-dynamics RK workspace exceeds maximum_workspace_bytes."
            )
        mean_indices = np.asarray(self.layout.mean_indices)
        eddy_indices = np.asarray(self.layout.eddy_indices)
        mean_to_eddy = np.asarray(self.dynamics.linear)[
            np.ix_(eddy_indices, mean_indices)
        ]
        constant_eddy = np.asarray(self.dynamics.constant)[eddy_indices]
        mean_mean_to_eddy = np.asarray(self.dynamics.quadratic)[
            np.ix_(eddy_indices, mean_indices, mean_indices)
        ]
        invariant_defect = max(
            float(np.max(np.abs(mean_to_eddy), initial=0.0)),
            float(np.max(np.abs(constant_eddy), initial=0.0)),
            float(np.max(np.abs(mean_mean_to_eddy), initial=0.0)),
        )
        if invariant_defect > self.hermitian_tolerance:
            raise ValueError(
                "The declared low/mean subspace is not invariant under constant, linear, "
                "and low-low dynamics; exact QL/GQL closure is unavailable."
            )
        cost = StatisticalDynamicsCost(
            state_dimension=self.layout.state_size,
            mean_dimension=self.layout.mean_dimension,
            covariance_dimension=self.layout.eddy_dimension,
            dense_state_bytes=dense_state_bytes,
            dense_workspace_bytes=dense_workspace_bytes,
            maximum_state_bytes=self.maximum_state_bytes,
            maximum_workspace_bytes=self.maximum_workspace_bytes,
        )
        return PreparedStatisticalDynamics(
            self,
            cost,
            invariant_defect=jnp.asarray(invariant_defect),
        )


class DenseCumulantTendency(StrictModule):
    mean: Array
    covariance: Array


class StatisticalStepEvidence(StrictModule):
    input: CumulantStateEvidence
    output: CumulantStateEvidence
    finite: Array
    accepted: Array
    time: Array
    step: Array
    prepared_id: str = eqx.field(static=True)


class StatisticalStepResult(StrictModule):
    state: CumulantState
    pre_truncation_state: DenseCumulantState
    evidence: StatisticalStepEvidence
    rank_event: RankAdaptationEvent | None


class PreparedStatisticalDynamics(StrictModule, NonTrainableState):
    plan: StatisticalDynamicsPlan
    cost: StatisticalDynamicsCost
    invariant_defect: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: StatisticalDynamicsPlan,
        cost: StatisticalDynamicsCost,
        /,
        *,
        invariant_defect: ArrayLike,
    ):
        if not isinstance(plan, StatisticalDynamicsPlan) or not isinstance(
            cost, StatisticalDynamicsCost
        ):
            raise TypeError("Prepared statistical dynamics require their plan and cost.")
        self.plan = plan
        self.cost = cost
        self.invariant_defect = jnp.asarray(invariant_defect)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-statistical-dynamics",
                "plan": plan.plan_id,
                "cost": {
                    "state": cost.dense_state_bytes,
                    "workspace": cost.dense_workspace_bytes,
                },
                "exactness": f"{plan.closure}-exact-for-{plan.interaction_model}",
            }
        )

    def _rhs_arrays(self, mean: Array, covariance: Array, /) -> DenseCumulantTendency:
        layout = self.plan.layout
        dynamics = self.plan.dynamics
        full_mean = layout.embed_mean(mean)
        full_covariance = layout.embed_covariance(covariance)
        mean_rhs = layout.restrict_mean(
            dynamics(full_mean) + dynamics.covariance_feedback(full_covariance)
        )
        jacobian = dynamics.jacobian(full_mean)
        eddy_jacobian = jacobian[jnp.ix_(layout.eddy_indices, layout.eddy_indices)]
        covariance_rhs = (
            oe.contract("ij,jk->ik", eddy_jacobian, covariance)
            + oe.contract("ij,jk->ik", covariance, _adjoint(eddy_jacobian))
            + self.plan.forcing.covariance
        )
        return DenseCumulantTendency(mean=mean_rhs, covariance=covariance_rhs)

    def rhs(self, state: CumulantState, /) -> DenseCumulantTendency:
        dense = densify_cumulant(state)
        require_valid_state(
            self.plan.layout,
            dense,
            hermitian_tolerance=self.plan.hermitian_tolerance,
            psd_tolerance=self.plan.psd_tolerance,
        )
        return self._rhs_arrays(dense.mean, dense.covariance)

    def _rk4(self, state: DenseCumulantState, step_size: Array, /) -> DenseCumulantState:
        mean = state.mean
        covariance = state.covariance
        half = 0.5 * step_size
        k1 = self._rhs_arrays(mean, covariance)
        k2 = self._rhs_arrays(
            mean + half * k1.mean,
            covariance + half * k1.covariance,
        )
        k3 = self._rhs_arrays(
            mean + half * k2.mean,
            covariance + half * k2.covariance,
        )
        k4 = self._rhs_arrays(
            mean + step_size * k3.mean,
            covariance + step_size * k3.covariance,
        )
        next_mean = mean + (step_size / 6.0) * (
            k1.mean + 2.0 * k2.mean + 2.0 * k3.mean + k4.mean
        )
        next_covariance = covariance + (step_size / 6.0) * (
            k1.covariance + 2.0 * k2.covariance + 2.0 * k3.covariance + k4.covariance
        )
        return DenseCumulantState(
            next_mean,
            next_covariance,
            layout_id=self.plan.layout.layout_id,
        )

    def step(
        self,
        state: CumulantState,
        /,
        *,
        time: ArrayLike = 0.0,
        step: ArrayLike = 0,
        step_size: ArrayLike | None = None,
        rank_policy: RankAdaptationPolicy | None = None,
    ) -> StatisticalStepResult:
        dense = densify_cumulant(state)
        input_evidence = require_valid_state(
            self.plan.layout,
            state,
            hermitian_tolerance=self.plan.hermitian_tolerance,
            psd_tolerance=self.plan.psd_tolerance,
            maximum_rank=(None if rank_policy is None else rank_policy.maximum_rank),
        )
        dt = jnp.asarray(
            self.plan.time_step if step_size is None else step_size,
            dtype=dense.mean.real.dtype,
        )
        if dt.shape != () or not bool(np.asarray(jnp.isfinite(dt) & (dt > 0.0))):
            raise ValueError("step_size must be a finite positive scalar.")
        pre_truncation = self._rk4(dense, dt)
        require_valid_state(
            self.plan.layout,
            pre_truncation,
            hermitian_tolerance=self.plan.hermitian_tolerance,
            psd_tolerance=self.plan.psd_tolerance,
        )
        rank_event: RankAdaptationEvent | None = None
        output: CumulantState = pre_truncation
        if isinstance(state, FactorCumulantState):
            if rank_policy is None:
                raise ValueError("Factor execution requires an explicit rank_policy.")
            adapted = factorize_cumulant(
                self.plan.layout,
                pre_truncation,
                rank_policy,
                previous_rank=state.rank,
                hermitian_tolerance=self.plan.hermitian_tolerance,
                psd_tolerance=self.plan.psd_tolerance,
            )
            output = adapted.state
            rank_event = adapted.event
        elif rank_policy is not None:
            raise ValueError("rank_policy is only meaningful for factor execution.")
        output_evidence = state_evidence(
            self.plan.layout,
            output,
            hermitian_tolerance=self.plan.hermitian_tolerance,
            psd_tolerance=self.plan.psd_tolerance,
            maximum_rank=(None if rank_policy is None else rank_policy.maximum_rank),
        )
        time_ = jnp.asarray(time, dtype=dt.dtype) + dt
        step_ = jnp.asarray(step, dtype=jnp.int32) + 1
        finite = output_evidence.finite & jnp.isfinite(time_)
        accepted = finite & output_evidence.successful
        evidence = StatisticalStepEvidence(
            input=input_evidence,
            output=output_evidence,
            finite=finite,
            accepted=accepted,
            time=time_,
            step=step_,
            prepared_id=self.prepared_id,
        )
        return StatisticalStepResult(
            state=output,
            pre_truncation_state=pre_truncation,
            evidence=evidence,
            rank_event=rank_event,
        )

    def execute(
        self,
        initial_state: CumulantState,
        steps: int,
        /,
        *,
        initial_time: ArrayLike = 0.0,
        rank_policy: RankAdaptationPolicy | None = None,
    ) -> "StatisticalDynamicsResult":
        count = int(steps)
        if count < 0:
            raise ValueError("steps must be non-negative.")
        state = initial_state
        time = jnp.asarray(initial_time)
        evidence: list[StatisticalStepEvidence] = []
        rank_events: list[RankAdaptationEvent] = []
        for step_index in range(count):
            result = self.step(
                state,
                time=time,
                step=step_index,
                rank_policy=rank_policy,
            )
            if not bool(np.asarray(result.evidence.accepted)):
                raise ValueError("Statistical-dynamics step failed its state gates.")
            state = result.state
            time = result.evidence.time
            evidence.append(result.evidence)
            if result.rank_event is not None:
                rank_events.append(result.rank_event)
        return StatisticalDynamicsResult(
            state=state,
            time=time,
            steps=jnp.asarray(count, dtype=jnp.int32),
            evidence=tuple(evidence),
            rank_events=tuple(rank_events),
            successful=jnp.asarray(
                all(bool(np.asarray(item.accepted)) for item in evidence)
            ),
            prepared_id=self.prepared_id,
        )

    def checkpoint(
        self,
        state: CumulantState,
        time: ArrayLike,
        step: int,
        /,
    ) -> "StatisticalDynamicsCheckpoint":
        require_valid_state(
            self.plan.layout,
            state,
            hermitian_tolerance=self.plan.hermitian_tolerance,
            psd_tolerance=self.plan.psd_tolerance,
        )
        time_ = jnp.asarray(time)
        step_ = int(step)
        if time_.shape != () or not bool(np.asarray(jnp.isfinite(time_))) or step_ < 0:
            raise ValueError("Checkpoint time and step are invalid.")
        checkpoint_id = canonical_fingerprint(
            {
                "kind": "statistical-dynamics-checkpoint",
                "prepared": self.prepared_id,
                "time": float(np.asarray(time_)),
                "step": step_,
                "state": array_tree_fingerprint(state),
            }
        )
        return StatisticalDynamicsCheckpoint(
            state=state,
            time=time_,
            step=jnp.asarray(step_, dtype=jnp.int32),
            prepared_id=self.prepared_id,
            checkpoint_id=checkpoint_id,
        )

    def restart(
        self,
        checkpoint: "StatisticalDynamicsCheckpoint",
        /,
    ) -> tuple[CumulantState, Array, Array]:
        if not isinstance(checkpoint, StatisticalDynamicsCheckpoint):
            raise TypeError("checkpoint must be a StatisticalDynamicsCheckpoint.")
        if checkpoint.prepared_id != self.prepared_id:
            raise ValueError("Checkpoint belongs to another prepared dynamics owner.")
        require_valid_state(
            self.plan.layout,
            checkpoint.state,
            hermitian_tolerance=self.plan.hermitian_tolerance,
            psd_tolerance=self.plan.psd_tolerance,
        )
        return checkpoint.state, checkpoint.time, checkpoint.step


class StatisticalDynamicsCheckpoint(StrictModule):
    state: CumulantState
    time: Array
    step: Array
    prepared_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)


class StatisticalDynamicsResult(StrictModule):
    state: CumulantState
    time: Array
    steps: Array
    evidence: tuple[StatisticalStepEvidence, ...]
    rank_events: tuple[RankAdaptationEvent, ...]
    successful: Array
    prepared_id: str = eqx.field(static=True)


class StatisticalContinuationStageEvidence(StrictModule):
    coordinate: Array
    accepted: Array
    start_state_id: str = eqx.field(static=True)
    end_state_id: str = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class StatisticalContinuationResult(StrictModule):
    state: CumulantState
    stage_results: tuple[StatisticalDynamicsResult, ...]
    evidence: tuple[StatisticalContinuationStageEvidence, ...]
    completed: Array
    schedule_id: str = eqx.field(static=True)


def execute_interaction_continuation(
    schedule: InteractionContinuationSchedule,
    prepared_stages: Sequence[PreparedStatisticalDynamics],
    initial_state: CumulantState,
    /,
    *,
    steps_per_stage: int,
    rank_policy: RankAdaptationPolicy | None = None,
) -> StatisticalContinuationResult:
    if not isinstance(schedule, InteractionContinuationSchedule):
        raise TypeError("schedule must be an InteractionContinuationSchedule.")
    prepared = tuple(prepared_stages)
    if len(prepared) != len(schedule.stages) or any(
        not isinstance(item, PreparedStatisticalDynamics) for item in prepared
    ):
        raise ValueError(
            "One prepared statistical plan is required per continuation stage."
        )
    if any(
        item.plan.layout.layout_id != prepared[0].plan.layout.layout_id
        for item in prepared
    ):
        raise ValueError("Continuation stages must share one cumulant layout.")
    steps = int(steps_per_stage)
    if steps < 1:
        raise ValueError("steps_per_stage must be positive.")
    state = initial_state
    results: list[StatisticalDynamicsResult] = []
    evidence: list[StatisticalContinuationStageEvidence] = []
    for stage, owner in zip(schedule.stages, prepared, strict=True):
        start_id = canonical_fingerprint(array_tree_fingerprint(state))
        result = owner.execute(state, steps, rank_policy=rank_policy)
        end_id = canonical_fingerprint(array_tree_fingerprint(result.state))
        accepted = result.successful
        results.append(result)
        evidence.append(
            StatisticalContinuationStageEvidence(
                coordinate=jnp.asarray(stage.coordinate),
                accepted=accepted,
                start_state_id=start_id,
                end_state_id=end_id,
                stage_id=stage.stage_id,
                prepared_id=owner.prepared_id,
            )
        )
        if not bool(np.asarray(accepted)):
            break
        state = result.state
    completed = len(results) == len(schedule.stages) and all(
        bool(np.asarray(item.accepted)) for item in evidence
    )
    return StatisticalContinuationResult(
        state=state,
        stage_results=tuple(results),
        evidence=tuple(evidence),
        completed=jnp.asarray(completed),
        schedule_id=schedule.schedule_id,
    )


def interaction_continuation_problem(
    residual: Callable[[Any, Array, Any], Any],
    /,
    *,
    problem_id: str,
) -> ParameterContinuationProblem:
    """Bind an interaction coordinate in ``[0, 1]`` to native continuation."""
    if not callable(residual):
        raise TypeError("residual must be callable.")
    identifier = str(problem_id)
    if not identifier:
        raise ValueError("problem_id must be non-empty.")
    return ParameterContinuationProblem(
        residual,
        parameter_lower=0.0,
        parameter_upper=1.0,
        problem_id=canonical_fingerprint(
            {"kind": "interaction-coordinate-continuation", "owner": identifier}
        ),
    )


__all__ = [
    "ClosureKind",
    "DenseCumulantTendency",
    "PreparedStatisticalDynamics",
    "QuadraticDynamics",
    "StatisticalContinuationResult",
    "StatisticalContinuationStageEvidence",
    "StatisticalDynamicsCheckpoint",
    "StatisticalDynamicsCost",
    "StatisticalDynamicsPlan",
    "StatisticalDynamicsResult",
    "StatisticalStepEvidence",
    "StatisticalStepResult",
    "execute_interaction_continuation",
    "interaction_continuation_problem",
]
