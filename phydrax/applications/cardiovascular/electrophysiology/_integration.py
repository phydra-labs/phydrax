#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared physical monodomain storage and operator-split integration.

The kernel equation uses positive-outward diffusion and ionic currents,

``C_i dV_i/dt + (K V)_i + volume_i I_ion,i = volume_i I_inward,i``.

Here voltage is in mV, time in ms, lumped capacitance in uF, ``K V`` in uA,
and volumetric currents in uA/mm3.  Thus every term above is exactly uA.
Tensor construction and spatial discretization are intentionally outside this
module; callers bind either a public linear operator or a generic tensor-
diffusion action together with its already-discretized public operator.
"""

from __future__ import annotations

from enum import IntFlag
from math import isfinite
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....equations import TensorDiffusionAction
from ....linalg import (
    AbstractLinearOperator,
    ArraySpace,
    AutoLinearMethod,
    DiagonalLinearOperator,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    OperatorProperties,
    prepare as prepare_linear_solve,
    PreparedLinearSolve,
    solve as solve_linear_system,
)
from ._reaction import PreparedReaction
from ._regional_assignment import PreparedRegionalAssignment


class PhysicalMonodomainStatus(IntFlag):
    """Fail-closed bitwise status for one physical macro step."""

    SUCCESS = 0
    NONFINITE = 1
    DIFFUSION_SOLVE_FAILURE = 2
    DIFFUSION_RESIDUAL_FAILURE = 4
    REACTION_FAILURE = 8
    INVALID_INPUT = 16
    EVENT_MISALIGNMENT = 32


class LieSplit(StrictModule, NonTrainableState):
    """First-order reaction-then-diffusion Lie composition."""

    split_id: str = eqx.field(static=True)

    def __init__(self):
        self.split_id = "cardiovascular-lie-reaction-diffusion-v1"


class StrangSplit(StrictModule, NonTrainableState):
    """Second-order diffusion-reaction-diffusion Strang composition."""

    split_id: str = eqx.field(static=True)

    def __init__(self):
        self.split_id = "cardiovascular-strang-diffusion-reaction-v1"


MonodomainSplitting: TypeAlias = LieSplit | StrangSplit


class ExplicitReferenceDiffusion(StrictModule, NonTrainableState):
    """Forward-Euler diffusion reference with an explicit certified step bound."""

    maximum_step_ms: float = eqx.field(static=True)
    diffusion_id: str = eqx.field(static=True)

    def __init__(self, maximum_step_ms: float, /):
        if isinstance(maximum_step_ms, bool):
            raise TypeError("maximum_step_ms must be a real scalar, not bool.")
        maximum = float(maximum_step_ms)
        if not isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_step_ms must be finite and positive.")
        self.maximum_step_ms = maximum
        self.diffusion_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-explicit-reference-diffusion-v1",
                "maximum_step_ms": maximum,
            }
        )


class ImplicitThetaDiffusion(StrictModule, NonTrainableState):
    """Implicit theta diffusion with an explicit PhydraX linear policy."""

    theta: float = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    diffusion_id: str = eqx.field(static=True)

    def __init__(self, theta: float, linear_policy: LinearSolvePolicy, /):
        if isinstance(theta, bool):
            raise TypeError("theta must be a real scalar, not bool.")
        theta_ = float(theta)
        if not isfinite(theta_) or not 0.5 <= theta_ <= 1.0:
            raise ValueError("theta must be finite and lie in [0.5, 1].")
        if not isinstance(linear_policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a LinearSolvePolicy.")
        if isinstance(linear_policy.method, AutoLinearMethod):
            raise ValueError(
                "Implicit monodomain diffusion requires an explicit linear method."
            )
        self.theta = theta_
        self.linear_policy = linear_policy
        self.diffusion_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-implicit-theta-diffusion-v1",
                "theta": theta_,
                "linear_method": linear_policy.method.name,
                "relative_tolerance": linear_policy.tolerance.relative,
                "absolute_tolerance": linear_policy.tolerance.absolute,
                "maximum_steps": linear_policy.tolerance.max_steps,
            }
        )


MonodomainDiffusionMethod: TypeAlias = ExplicitReferenceDiffusion | ImplicitThetaDiffusion


class EventAlignedMultirateSchedule(StrictModule, NonTrainableState):
    """Integer-tick reaction/diffusion/checkpoint cadence with pinned events."""

    tick_dt_ms: float = eqx.field(static=True)
    reaction_ticks_per_macro: int = eqx.field(static=True)
    macro_step_count: int = eqx.field(static=True)
    event_ticks: tuple[int, ...] = eqx.field(static=True)
    checkpoint_stride: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        tick_dt_ms: float,
        reaction_ticks_per_macro: int,
        macro_step_count: int,
        /,
        *,
        event_ticks: tuple[int, ...] = (0,),
        checkpoint_stride: int = 1,
    ):
        if isinstance(tick_dt_ms, bool):
            raise TypeError("tick_dt_ms must be a real scalar, not bool.")
        tick = float(tick_dt_ms)
        if not isfinite(tick) or tick <= 0.0:
            raise ValueError("tick_dt_ms must be finite and positive.")
        integer_values = (
            reaction_ticks_per_macro,
            macro_step_count,
            checkpoint_stride,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integer_values
        ):
            raise TypeError("Schedule cadence values must be integers.")
        if any(value <= 0 for value in integer_values):
            raise ValueError("Schedule cadence values must be positive.")
        events = tuple(int(value) for value in event_ticks)
        total_ticks = reaction_ticks_per_macro * macro_step_count
        if not events or events[0] != 0:
            raise ValueError("event_ticks must start at zero.")
        if tuple(sorted(set(events))) != events:
            raise ValueError("event_ticks must be strictly increasing and unique.")
        if events[-1] >= total_ticks:
            raise ValueError("event_ticks must lie inside the fixed schedule horizon.")
        self.tick_dt_ms = tick
        self.reaction_ticks_per_macro = reaction_ticks_per_macro
        self.macro_step_count = macro_step_count
        self.event_ticks = events
        self.checkpoint_stride = checkpoint_stride
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-event-aligned-multirate-v1",
                "tick_dt_ms": tick,
                "reaction_ticks_per_macro": reaction_ticks_per_macro,
                "macro_step_count": macro_step_count,
                "event_ticks": events,
                "checkpoint_stride": checkpoint_stride,
            }
        )

    @property
    def macro_dt_ms(self) -> float:
        return self.tick_dt_ms * self.reaction_ticks_per_macro

    @property
    def total_tick_count(self) -> int:
        return self.reaction_ticks_per_macro * self.macro_step_count

    def event_mask(self) -> Array:
        mask = jnp.zeros((self.total_tick_count,), dtype=bool)
        return mask.at[jnp.asarray(self.event_ticks, dtype=jnp.int32)].set(True)


class PublicDiffusionOperatorInput(StrictModule, NonTrainableState):
    """Already-discretized positive-outward diffusion operator input."""

    operator: AbstractLinearOperator
    regional_assignment_id: str = eqx.field(static=True)
    input_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        regional_assignment_id: str,
        /,
        *,
        input_id: str,
    ):
        _validate_diffusion_operator(operator)
        self.operator = operator
        self.regional_assignment_id = _identifier(
            regional_assignment_id, "regional_assignment_id"
        )
        self.input_id = _identifier(input_id, "input_id")


class TensorDiffusionOperatorInput(StrictModule, NonTrainableState):
    """Generic tensor action plus its external fixed-topology discretization."""

    tensor_diffusion_action: TensorDiffusionAction
    operator: AbstractLinearOperator
    tensor_action_id: str = eqx.field(static=True)
    regional_assignment_id: str = eqx.field(static=True)
    input_id: str = eqx.field(static=True)

    def __init__(
        self,
        tensor_diffusion_action: TensorDiffusionAction,
        operator: AbstractLinearOperator,
        regional_assignment_id: str,
        /,
        *,
        tensor_action_id: str | None = None,
        input_id: str,
    ):
        if not isinstance(tensor_diffusion_action, TensorDiffusionAction):
            raise TypeError("tensor_diffusion_action must be a TensorDiffusionAction.")
        _validate_diffusion_operator(operator)
        self.tensor_diffusion_action = tensor_diffusion_action
        self.operator = operator
        action_id = tensor_diffusion_action.action_id
        if (
            tensor_action_id is not None
            and _identifier(tensor_action_id, "tensor_action_id") != action_id
        ):
            raise ValueError(
                "tensor_action_id must match TensorDiffusionAction.action_id."
            )
        self.tensor_action_id = action_id
        self.regional_assignment_id = _identifier(
            regional_assignment_id, "regional_assignment_id"
        )
        self.input_id = _identifier(input_id, "input_id")


DiffusionOperatorInput: TypeAlias = (
    PublicDiffusionOperatorInput | TensorDiffusionOperatorInput
)


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _validate_diffusion_operator(operator: AbstractLinearOperator, /) -> None:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if not isinstance(operator.source, ArraySpace) or not isinstance(
        operator.target, ArraySpace
    ):
        raise TypeError("Monodomain diffusion requires ArraySpace source and target.")
    if operator.source.shape != operator.target.shape or len(operator.source.shape) != 1:
        raise ValueError("Monodomain diffusion must be a square vector operator.")
    if operator.batch_shape:
        raise ValueError("Monodomain diffusion must be unbatched.")
    if operator.properties.self_adjoint is not True:
        raise ValueError("Diffusion operator must certify self-adjointness.")
    if operator.properties.positive_semidefinite is not True:
        raise ValueError("Diffusion operator must certify positive semidefiniteness.")
    if np.dtype(operator.source.dtype).kind != "f":
        raise TypeError("Monodomain diffusion coordinates must be real floating point.")


class PhysicalMonodomainSpatialBinding(StrictModule, NonTrainableState):
    """Nodal volumes and external diffusion discretization for one topology."""

    node_volume_mm3: Array
    diffusion: DiffusionOperatorInput
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        node_volume_mm3: ArrayLike,
        diffusion: DiffusionOperatorInput,
        /,
        *,
        binding_id: str | None = None,
    ):
        volumes = jnp.asarray(node_volume_mm3)
        if volumes.ndim != 1:
            raise ValueError("node_volume_mm3 must be a vector.")
        if not jnp.issubdtype(volumes.dtype, jnp.floating):
            raise TypeError("node_volume_mm3 must have floating dtype.")
        volumes = eqx.error_if(
            volumes,
            jnp.any(~jnp.isfinite(volumes)) | jnp.any(volumes <= 0.0),
            "node_volume_mm3 must be finite and positive.",
        )
        if not isinstance(
            diffusion, (PublicDiffusionOperatorInput, TensorDiffusionOperatorInput)
        ):
            raise TypeError("diffusion must be a supported diffusion operator input.")
        if diffusion.operator.source.shape != volumes.shape:
            raise ValueError("Diffusion operator size must match node_volume_mm3.")
        if np.dtype(diffusion.operator.source.dtype) != np.dtype(volumes.dtype):
            raise TypeError("Diffusion operator and node volumes must use one dtype.")
        label = None if binding_id is None else _identifier(binding_id, "binding_id")
        tensor_payload = (
            None
            if isinstance(diffusion, PublicDiffusionOperatorInput)
            else {
                "action_id": diffusion.tensor_diffusion_action.action_id,
                "field_name": diffusion.tensor_diffusion_action.field_name,
                "tensor_axes": diffusion.tensor_diffusion_action.tensor_axes,
                "content": array_tree_fingerprint(diffusion.tensor_diffusion_action),
            }
        )
        self.node_volume_mm3 = volumes
        self.diffusion = diffusion
        self.binding_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-physical-monodomain-spatial-binding-v1",
                "label": label,
                "node_volume_mm3": array_tree_fingerprint(volumes),
                "diffusion_input_id": diffusion.input_id,
                "operator_id": diffusion.operator.operator_id,
                "operator_content": array_tree_fingerprint(diffusion.operator),
                "regional_assignment_id": diffusion.regional_assignment_id,
                "tensor_diffusion_action": tensor_payload,
            }
        )


class PhysicalMonodomainPlan(StrictModule, NonTrainableState):
    """Immutable integration plan, distinct from phenomenological fidelity plans."""

    node_count: int = eqx.field(static=True)
    schedule: EventAlignedMultirateSchedule
    splitting: MonodomainSplitting
    diffusion: MonodomainDiffusionMethod
    residual_tolerance: float = eqx.field(static=True)
    checkpoint_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        node_count: int,
        schedule: EventAlignedMultirateSchedule,
        splitting: MonodomainSplitting,
        diffusion: MonodomainDiffusionMethod,
        /,
        *,
        residual_tolerance: float = 1.0e-7,
        checkpoint_capacity: int = 2,
    ):
        if not isinstance(node_count, int) or isinstance(node_count, bool):
            raise TypeError("node_count must be an integer.")
        if node_count <= 0:
            raise ValueError("node_count must be positive.")
        if not isinstance(schedule, EventAlignedMultirateSchedule):
            raise TypeError("schedule must be an EventAlignedMultirateSchedule.")
        if not isinstance(splitting, (LieSplit, StrangSplit)):
            raise TypeError("splitting must be LieSplit or StrangSplit.")
        if not isinstance(
            diffusion, (ExplicitReferenceDiffusion, ImplicitThetaDiffusion)
        ):
            raise TypeError(
                "diffusion must be ExplicitReferenceDiffusion or ImplicitThetaDiffusion."
            )
        if isinstance(residual_tolerance, bool):
            raise TypeError("residual_tolerance must be a real scalar, not bool.")
        tolerance = float(residual_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("residual_tolerance must be finite and positive.")
        if not isinstance(checkpoint_capacity, int) or isinstance(
            checkpoint_capacity, bool
        ):
            raise TypeError("checkpoint_capacity must be an integer.")
        if checkpoint_capacity <= 0:
            raise ValueError("checkpoint_capacity must be positive.")
        diffusion_stage_dt = (
            0.5 * schedule.macro_dt_ms
            if isinstance(splitting, StrangSplit)
            else schedule.macro_dt_ms
        )
        if (
            isinstance(diffusion, ExplicitReferenceDiffusion)
            and diffusion_stage_dt > diffusion.maximum_step_ms
        ):
            raise ValueError(
                "Explicit reference diffusion stage exceeds maximum_step_ms."
            )
        self.node_count = node_count
        self.schedule = schedule
        self.splitting = splitting
        self.diffusion = diffusion
        self.residual_tolerance = tolerance
        self.checkpoint_capacity = checkpoint_capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-physical-monodomain-plan-v1",
                "node_count": node_count,
                "schedule": schedule.schedule_id,
                "splitting": splitting.split_id,
                "diffusion": diffusion.diffusion_id,
                "residual_tolerance": tolerance,
                "checkpoint_capacity": checkpoint_capacity,
            }
        )

    def prepare(
        self,
        spatial: PhysicalMonodomainSpatialBinding,
        regional_assignment: PreparedRegionalAssignment,
        reactions: tuple[PreparedReaction, ...],
        /,
    ) -> PreparedPhysicalMonodomain:
        return prepare_physical_monodomain(self, spatial, regional_assignment, reactions)


class HomogeneousReactionWorkset(StrictModule, NonTrainableState):
    """One fixed node index set sharing model, parameters, and regional scaling."""

    reaction: PreparedReaction
    node_indices: Array
    exact_gate_mask: Array
    capacitance_scale: float = eqx.field(static=True)
    ionic_current_scale: float = eqx.field(static=True)
    state_update_scale: float = eqx.field(static=True)
    volumetric_capacitance_uF_per_mm3: float = eqx.field(static=True)
    workset_id: str = eqx.field(static=True)


class PreparedPhysicalMonodomain(StrictModule, NonTrainableState):
    """Prepared fixed-topology physical monodomain integration runtime."""

    plan: PhysicalMonodomainPlan
    spatial: PhysicalMonodomainSpatialBinding
    regional_assignment: PreparedRegionalAssignment
    worksets: tuple[HomogeneousReactionWorkset, ...]
    lumped_capacitance_uF: Array
    full_diffusion_solve: PreparedLinearSolve | None
    half_diffusion_solve: PreparedLinearSolve | None
    event_mask: Array
    runtime_id: str = eqx.field(static=True)


class MonodomainCheckpointBuffer(StrictModule):
    """Fixed-capacity ring retaining complete accepted runtime states."""

    voltage_mV: Array
    local_states: tuple[Array, ...]
    tick: Array
    macro_step_index: Array
    last_applied_inward_current_uA_per_mm3: Array
    has_previous_stimulus: Array
    valid: Array
    write_cursor: Array
    runtime_id: str = eqx.field(static=True)


class PhysicalMonodomainState(StrictModule):
    """Complete physical state at an accepted macro boundary."""

    voltage_mV: Array
    local_states: tuple[Array, ...]
    tick: Array
    macro_step_index: Array
    last_applied_inward_current_uA_per_mm3: Array
    has_previous_stimulus: Array
    checkpoints: MonodomainCheckpointBuffer
    runtime_id: str = eqx.field(static=True)


class MonodomainMacroInputs(StrictModule):
    """Positive-inward volumetric current on every reaction tick."""

    inward_current_uA_per_mm3: Array


class ScheduledMonodomainInputs(StrictModule):
    """Fixed-horizon positive-inward current schedule."""

    inward_current_uA_per_mm3: Array


class MonodomainStepEvidence(StrictModule):
    """Original candidate evidence retained even when commit rolls back."""

    diffusion_residual_norm_uA: Array
    diffusion_relative_residual: Array
    diffusion_linear_status: Array
    diffusion_iterations: Array
    diffusion_stage_active: Array
    reaction_admissible: Array
    reaction_rate_call_count: Array
    exact_gate_call_count: Array
    workset_node_count: Array
    event_aligned: Array
    finite: Array
    start_tick: Array
    end_tick: Array
    reaction_tick_count: Array
    diffusion_stage_count: Array
    status: Array
    successful: Array
    rolled_back: Array
    checkpoint_due: Array
    checkpoint_written: Array


class PhysicalMonodomainCandidate(StrictModule):
    """Uncommitted state and its complete original physical evidence."""

    source_state: PhysicalMonodomainState
    state: PhysicalMonodomainState
    evidence: MonodomainStepEvidence


class PhysicalMonodomainStepResult(StrictModule):
    """Accepted or rolled-back state together with the original candidate."""

    state: PhysicalMonodomainState
    candidate: PhysicalMonodomainCandidate
    evidence: MonodomainStepEvidence


class MonodomainIntegrationResult(StrictModule):
    """Fixed-horizon accepted state, candidates, and per-step evidence."""

    state: PhysicalMonodomainState
    candidate_voltage_mV: Array
    evidence: MonodomainStepEvidence


class MonodomainRollbackEvidence(StrictModule):
    """Evidence for an explicit restoration from the latest checkpoint."""

    source_slot: Array
    source_tick: Array
    restored: Array


class MonodomainRollbackResult(StrictModule):
    state: PhysicalMonodomainState
    evidence: MonodomainRollbackEvidence


class _DiffusionStageResult(StrictModule):
    voltage_mV: Array
    residual_norm_uA: Array
    relative_residual: Array
    linear_status: Array
    iterations: Array
    successful: Array


def _build_worksets(
    assignment: PreparedRegionalAssignment,
    reactions: tuple[PreparedReaction, ...],
    dtype: np.dtype,
    /,
) -> tuple[HomogeneousReactionWorkset, ...]:
    reactions_ = tuple(reactions)
    if not reactions_ or not all(
        isinstance(reaction, PreparedReaction) for reaction in reactions_
    ):
        raise TypeError("reactions must contain PreparedReaction values.")
    worksets: list[HomogeneousReactionWorkset] = []
    for position, node_indices in enumerate(assignment.workset_indices):
        reaction_index = assignment.workset_reaction_indices[position]
        if reaction_index >= len(reactions_):
            raise ValueError("A workset reaction_index is out of range.")
        reaction = reactions_[reaction_index]
        if reaction.node_count != int(node_indices.shape[0]):
            raise ValueError(
                "Each PreparedReaction node_count must equal its homogeneous "
                "workset size."
            )
        if reaction.plan.dtype != dtype:
            raise TypeError("Reaction and diffusion dtypes must match exactly.")
        gate_indices = tuple(
            index - 1 for index in reaction.model.state_layout.gate_indices
        )
        exact_mask = jnp.zeros((reaction.gate_count,), dtype=bool)
        if gate_indices:
            exact_mask = exact_mask.at[jnp.asarray(gate_indices, dtype=jnp.int32)].set(
                True
            )
        capacitance_scale = assignment.workset_capacitance_scales[position]
        volumetric_capacitance = float(
            reaction.model.membrane_surface_to_volume_per_mm
        ) * float(reaction.model.membrane_capacitance_uF_per_mm2)
        if not isfinite(volumetric_capacitance) or volumetric_capacitance <= 0.0:
            raise ValueError(
                "Reaction membrane scaling must define positive finite "
                "volumetric capacitance."
            )
        ionic_current_scale = assignment.workset_ionic_current_scales[position]
        state_update_scale = assignment.workset_state_update_scales[position]
        workset_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-homogeneous-reaction-workset-v1",
                "regional_workset_id": assignment.workset_ids[position],
                "reaction_plan_id": reaction.plan_id,
                "reaction_model_id": reaction.model_id,
                "reaction_default_parameters": array_tree_fingerprint(
                    reaction.model.default_parameters
                ),
                "state_layout": reaction.model.state_layout.state_names,
                "gate_indices": reaction.model.state_layout.gate_indices,
                "volumetric_capacitance_uF_per_mm3": volumetric_capacitance,
                "capacitance_scale": capacitance_scale,
                "ionic_current_scale": ionic_current_scale,
                "state_update_scale": state_update_scale,
            }
        )
        worksets.append(
            HomogeneousReactionWorkset(
                reaction,
                node_indices,
                exact_mask,
                capacitance_scale,
                ionic_current_scale,
                state_update_scale,
                volumetric_capacitance,
                workset_id,
            )
        )
    return tuple(worksets)


def _prepare_theta_solve(
    capacitance_uF: Array,
    operator: AbstractLinearOperator,
    dt_ms: float,
    method: ImplicitThetaDiffusion,
    stage_name: str,
    /,
) -> PreparedLinearSolve:
    dt = jnp.asarray(dt_ms, dtype=capacitance_uF.dtype)
    theta = jnp.asarray(method.theta, dtype=capacitance_uF.dtype)
    mass_rate = DiagonalLinearOperator(
        capacitance_uF / dt,
        space=operator.source,
        properties=OperatorProperties(
            diagonal=True,
            self_adjoint=True,
            positive_definite=True,
            rank=operator.source.size,
            evidence={
                "diagonal": "construction",
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
                "rank": "construction",
            },
        ),
        operator_id=f"monodomain-capacitance-rate-{stage_name}",
    )
    system = LinearSystem(
        mass_rate + theta * operator,
        problem_id=f"monodomain-theta-system-{stage_name}",
    )
    return prepare_linear_solve(system, method.linear_policy)


def prepare_physical_monodomain(
    plan: PhysicalMonodomainPlan,
    spatial: PhysicalMonodomainSpatialBinding,
    regional_assignment: PreparedRegionalAssignment,
    reactions: tuple[PreparedReaction, ...],
    /,
) -> PreparedPhysicalMonodomain:
    """Bind regional reaction worksets and reusable diffusion solves."""
    if not isinstance(plan, PhysicalMonodomainPlan):
        raise TypeError("plan must be a PhysicalMonodomainPlan.")
    if not isinstance(spatial, PhysicalMonodomainSpatialBinding):
        raise TypeError("spatial must be a PhysicalMonodomainSpatialBinding.")
    if not isinstance(regional_assignment, PreparedRegionalAssignment):
        raise TypeError("regional_assignment must be a PreparedRegionalAssignment.")
    if spatial.node_volume_mm3.shape != (plan.node_count,):
        raise ValueError("Spatial binding size differs from the monodomain plan.")
    if regional_assignment.node_count != plan.node_count:
        raise ValueError("Regional assignment size differs from the monodomain plan.")
    if spatial.diffusion.regional_assignment_id != regional_assignment.runtime_id:
        raise ValueError(
            "Diffusion input must identify the exact prepared regional assignment."
        )
    dtype = np.dtype(spatial.node_volume_mm3.dtype)
    worksets = _build_worksets(regional_assignment, reactions, dtype)
    lumped_capacitance = jnp.zeros_like(spatial.node_volume_mm3)
    for workset in worksets:
        density = workset.volumetric_capacitance_uF_per_mm3 * workset.capacitance_scale
        lumped_capacitance = lumped_capacitance.at[workset.node_indices].set(
            density * spatial.node_volume_mm3[workset.node_indices]
        )
    lumped_capacitance = eqx.error_if(
        lumped_capacitance,
        jnp.any(~jnp.isfinite(lumped_capacitance)) | jnp.any(lumped_capacitance <= 0.0),
        "Prepared lumped capacitance must be finite and positive.",
    )

    full_solve: PreparedLinearSolve | None = None
    half_solve: PreparedLinearSolve | None = None
    if isinstance(plan.diffusion, ImplicitThetaDiffusion):
        full_solve = _prepare_theta_solve(
            lumped_capacitance,
            spatial.diffusion.operator,
            plan.schedule.macro_dt_ms,
            plan.diffusion,
            "full",
        )
        if isinstance(plan.splitting, StrangSplit):
            half_solve = _prepare_theta_solve(
                lumped_capacitance,
                spatial.diffusion.operator,
                0.5 * plan.schedule.macro_dt_ms,
                plan.diffusion,
                "half",
            )
    runtime_id = canonical_fingerprint(
        {
            "kind": "prepared-cardiovascular-physical-monodomain-v1",
            "plan": plan.plan_id,
            "spatial": spatial.binding_id,
            "regional_assignment": regional_assignment.runtime_id,
            "worksets": tuple(workset.workset_id for workset in worksets),
            "full_linear_plan": (None if full_solve is None else full_solve.plan.plan_id),
            "half_linear_plan": (None if half_solve is None else half_solve.plan.plan_id),
        }
    )
    return PreparedPhysicalMonodomain(
        plan,
        spatial,
        regional_assignment,
        worksets,
        lumped_capacitance,
        full_solve,
        half_solve,
        plan.schedule.event_mask(),
        runtime_id,
    )


def _empty_checkpoint_buffer(
    runtime: PreparedPhysicalMonodomain,
    voltage_mV: Array,
    local_states: tuple[Array, ...],
    /,
) -> MonodomainCheckpointBuffer:
    capacity = runtime.plan.checkpoint_capacity
    return MonodomainCheckpointBuffer(
        jnp.zeros((capacity, runtime.plan.node_count), dtype=voltage_mV.dtype)
        .at[0]
        .set(voltage_mV),
        tuple(
            jnp.zeros((capacity,) + local.shape, dtype=local.dtype).at[0].set(local)
            for local in local_states
        ),
        jnp.zeros((capacity,), dtype=jnp.int32),
        jnp.zeros((capacity,), dtype=jnp.int32),
        jnp.zeros((capacity, runtime.plan.node_count), dtype=voltage_mV.dtype),
        jnp.zeros((capacity,), dtype=bool),
        jnp.zeros((capacity,), dtype=bool).at[0].set(True),
        jnp.asarray(1 % capacity, dtype=jnp.int32),
        runtime.runtime_id,
    )


def initialize_physical_monodomain(
    runtime: PreparedPhysicalMonodomain,
    /,
    *,
    voltage_mV: ArrayLike | None = None,
) -> PhysicalMonodomainState:
    """Initialize every homogeneous reaction block and the checkpoint ring."""
    if not isinstance(runtime, PreparedPhysicalMonodomain):
        raise TypeError("runtime must be a PreparedPhysicalMonodomain.")
    voltage = jnp.zeros(
        (runtime.plan.node_count,), dtype=runtime.spatial.node_volume_mm3.dtype
    )
    local_states: list[Array] = []
    for workset in runtime.worksets:
        workset_voltage, local = workset.reaction.initialize()
        voltage = voltage.at[workset.node_indices].set(workset_voltage)
        local_states.append(local)
    if voltage_mV is not None:
        supplied = jnp.asarray(voltage_mV, dtype=voltage.dtype)
        if supplied.shape != voltage.shape:
            raise ValueError(f"voltage_mV must have shape {voltage.shape}.")
        voltage = supplied
    local_tuple = tuple(local_states)
    checkpoints = _empty_checkpoint_buffer(runtime, voltage, local_tuple)
    return PhysicalMonodomainState(
        voltage,
        local_tuple,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros_like(voltage),
        jnp.asarray(False),
        checkpoints,
        runtime.runtime_id,
    )


def zero_monodomain_macro_inputs(
    runtime: PreparedPhysicalMonodomain, /
) -> MonodomainMacroInputs:
    return MonodomainMacroInputs(
        jnp.zeros(
            (
                runtime.plan.schedule.reaction_ticks_per_macro,
                runtime.plan.node_count,
            ),
            dtype=runtime.spatial.node_volume_mm3.dtype,
        )
    )


def zero_scheduled_monodomain_inputs(
    runtime: PreparedPhysicalMonodomain, /
) -> ScheduledMonodomainInputs:
    return ScheduledMonodomainInputs(
        jnp.zeros(
            (
                runtime.plan.schedule.total_tick_count,
                runtime.plan.node_count,
            ),
            dtype=runtime.spatial.node_volume_mm3.dtype,
        )
    )


def _validate_state_structure(
    runtime: PreparedPhysicalMonodomain,
    state: PhysicalMonodomainState,
    /,
) -> PhysicalMonodomainState:
    if not isinstance(state, PhysicalMonodomainState):
        raise TypeError("state must be a PhysicalMonodomainState.")
    if state.runtime_id != runtime.runtime_id:
        raise ValueError("State belongs to a different prepared monodomain runtime.")
    if state.checkpoints.runtime_id != runtime.runtime_id:
        raise ValueError("Checkpoint buffer belongs to a different monodomain runtime.")
    node_count = runtime.plan.node_count
    capacity = runtime.plan.checkpoint_capacity
    if state.voltage_mV.shape != (node_count,):
        raise ValueError("State voltage shape differs from the prepared topology.")
    if state.tick.shape != () or state.macro_step_index.shape != ():
        raise ValueError("State tick and macro_step_index must be scalars.")
    if state.last_applied_inward_current_uA_per_mm3.shape != (node_count,):
        raise ValueError("State stimulus history shape differs from the runtime.")
    if state.has_previous_stimulus.shape != ():
        raise ValueError("State stimulus-history flag must be scalar.")
    if len(state.local_states) != len(runtime.worksets):
        raise ValueError("State local workset count differs from the runtime.")
    if len(state.checkpoints.local_states) != len(runtime.worksets):
        raise ValueError("Checkpoint local workset count differs from the runtime.")
    checkpoint_shapes = (
        state.checkpoints.voltage_mV.shape == (capacity, node_count),
        state.checkpoints.tick.shape == (capacity,),
        state.checkpoints.macro_step_index.shape == (capacity,),
        state.checkpoints.last_applied_inward_current_uA_per_mm3.shape
        == (capacity, node_count),
        state.checkpoints.has_previous_stimulus.shape == (capacity,),
        state.checkpoints.valid.shape == (capacity,),
        state.checkpoints.write_cursor.shape == (),
    )
    if not all(checkpoint_shapes):
        raise ValueError("Checkpoint buffer shapes differ from the runtime.")
    for local, stored, workset in zip(
        state.local_states,
        state.checkpoints.local_states,
        runtime.worksets,
        strict=True,
    ):
        expected = (workset.node_indices.shape[0], workset.reaction.gate_count)
        if local.shape != expected or stored.shape != (capacity,) + expected:
            raise ValueError(
                "A local reaction-state or checkpoint shape differs from its workset."
            )
    checked_cursor = eqx.error_if(
        state.checkpoints.write_cursor,
        (state.checkpoints.write_cursor < 0)
        | (state.checkpoints.write_cursor >= capacity),
        "Checkpoint write cursor lies outside its fixed capacity.",
    )
    return eqx.tree_at(
        lambda value: value.checkpoints.write_cursor,
        state,
        checked_cursor,
    )


def _validate_state(
    runtime: PreparedPhysicalMonodomain,
    state: PhysicalMonodomainState,
    /,
) -> PhysicalMonodomainState:
    state = _validate_state_structure(runtime, state)
    stride = runtime.plan.schedule.reaction_ticks_per_macro
    invalid_cadence = (
        (state.tick < 0)
        | (state.macro_step_index < 0)
        | (state.macro_step_index > runtime.plan.schedule.macro_step_count)
        | (state.tick > runtime.plan.schedule.total_tick_count)
        | (state.tick != state.macro_step_index * stride)
    )
    checked_tick = eqx.error_if(
        state.tick,
        invalid_cadence,
        "Monodomain state violates its fixed integer cadence or horizon.",
    )
    return eqx.tree_at(lambda value: value.tick, state, checked_tick)


def _require_step_available(
    runtime: PreparedPhysicalMonodomain,
    state: PhysicalMonodomainState,
    /,
) -> PhysicalMonodomainState:
    checked_voltage = eqx.error_if(
        state.voltage_mV,
        state.macro_step_index >= runtime.plan.schedule.macro_step_count,
        "Monodomain state is already at the fixed schedule horizon.",
    )
    return eqx.tree_at(lambda value: value.voltage_mV, state, checked_voltage)


def _states_exactly_equal(
    left: PhysicalMonodomainState,
    right: PhysicalMonodomainState,
    /,
) -> Array:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    if len(left_leaves) != len(right_leaves):
        return jnp.asarray(False)
    equal = jnp.asarray(True)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if left_leaf.shape != right_leaf.shape or left_leaf.dtype != right_leaf.dtype:
            return jnp.asarray(False)
        equal = equal & jnp.array_equal(left_leaf, right_leaf, equal_nan=True)
    return equal


def _macro_inputs(
    runtime: PreparedPhysicalMonodomain,
    inputs: MonodomainMacroInputs,
    /,
) -> Array:
    if not isinstance(inputs, MonodomainMacroInputs):
        raise TypeError("inputs must be MonodomainMacroInputs.")
    current = jnp.asarray(inputs.inward_current_uA_per_mm3)
    expected = (
        runtime.plan.schedule.reaction_ticks_per_macro,
        runtime.plan.node_count,
    )
    if current.shape != expected:
        raise ValueError(f"inward_current_uA_per_mm3 must have shape {expected}.")
    if current.dtype != runtime.spatial.node_volume_mm3.dtype:
        raise TypeError("Monodomain inputs must use the prepared runtime dtype.")
    return current


def _event_alignment(
    runtime: PreparedPhysicalMonodomain,
    state: PhysicalMonodomainState,
    current: Array,
    /,
) -> Array:
    previous = jnp.concatenate(
        (state.last_applied_inward_current_uA_per_mm3[None, :], current[:-1]),
        axis=0,
    )
    changed = jnp.any(current != previous, axis=1)
    changed = changed.at[0].set(changed[0] | ~state.has_previous_stimulus)
    indices = state.tick + jnp.arange(current.shape[0], dtype=jnp.int32)
    inside = indices < runtime.event_mask.shape[0]
    clipped = jnp.minimum(indices, runtime.event_mask.shape[0] - 1)
    allowed = inside & runtime.event_mask[clipped]
    return jnp.all(~changed | allowed)


def _reaction_advance(
    runtime: PreparedPhysicalMonodomain,
    voltage_mV: Array,
    local_states: tuple[Array, ...],
    inward_current: Array,
    /,
) -> tuple[Array, tuple[Array, ...], Array]:
    dt = jnp.asarray(runtime.plan.schedule.tick_dt_ms, dtype=voltage_mV.dtype)

    def tick_body(carry, stimulus):
        voltage, locals_ = carry
        updated_locals: list[Array] = []
        for local, workset in zip(locals_, runtime.worksets, strict=True):
            indices = workset.node_indices
            local_voltage = voltage[indices]
            local_stimulus = stimulus[indices]
            ionic_voltage_rate, local_rate = workset.reaction.rates(
                local_voltage, local, 0.0
            )
            capacitance_scale = jnp.asarray(
                workset.capacitance_scale, dtype=voltage.dtype
            )
            ionic_scale = jnp.asarray(workset.ionic_current_scale, dtype=voltage.dtype)
            state_scale = jnp.asarray(workset.state_update_scale, dtype=voltage.dtype)
            applied_voltage_rate = local_stimulus / (
                workset.volumetric_capacitance_uF_per_mm3 * capacitance_scale
            )
            next_voltage = local_voltage + dt * (
                ionic_scale * ionic_voltage_rate / capacitance_scale
                + applied_voltage_rate
            )
            euler_local = local + dt * state_scale * local_rate
            exact_local = workset.reaction.exact_gate_update(local_voltage, local, dt)
            exact_local = local + state_scale * (exact_local - local)
            next_local = jnp.where(
                workset.exact_gate_mask[None, :], exact_local, euler_local
            )
            voltage = voltage.at[indices].set(next_voltage)
            updated_locals.append(next_local)
        return (voltage, tuple(updated_locals)), None

    (updated_voltage, updated_local), _ = jax.lax.scan(
        tick_body, (voltage_mV, local_states), inward_current
    )
    admissible = jnp.asarray(True)
    for local, workset in zip(updated_local, runtime.worksets, strict=True):
        admissible = admissible & jnp.all(
            workset.reaction.admissible(updated_voltage[workset.node_indices], local)
        )
    return updated_voltage, updated_local, admissible


def _diffusion_advance(
    runtime: PreparedPhysicalMonodomain,
    voltage_mV: Array,
    dt_ms: float,
    prepared_solve: PreparedLinearSolve | None,
    /,
) -> _DiffusionStageResult:
    operator = runtime.spatial.diffusion.operator
    capacitance = runtime.lumped_capacitance_uF
    dt = jnp.asarray(dt_ms, dtype=voltage_mV.dtype)
    old_action = operator.mv(voltage_mV)
    if isinstance(runtime.plan.diffusion, ExplicitReferenceDiffusion):
        candidate = voltage_mV - dt * old_action / capacitance
        linear_status = jnp.asarray(int(LinearSolveStatus.SUCCESS), dtype=jnp.int32)
        iterations = jnp.asarray(0, dtype=jnp.int32)
        theta = jnp.asarray(0.0, dtype=voltage_mV.dtype)
        linear_successful = jnp.asarray(True)
    else:
        if prepared_solve is None:
            raise ValueError("Implicit diffusion requires its prepared linear solve.")
        theta = jnp.asarray(runtime.plan.diffusion.theta, dtype=voltage_mV.dtype)
        right = capacitance * voltage_mV / dt - (1.0 - theta) * old_action
        solve_result = solve_linear_system(prepared_solve, right)
        candidate = solve_result.value
        linear_status = solve_result.status
        iterations = solve_result.diagnostics.iterations
        linear_successful = solve_result.successful
    theta_voltage = theta * candidate + (1.0 - theta) * voltage_mV
    residual = capacitance * (candidate - voltage_mV) / dt + operator.mv(theta_voltage)
    residual_norm = jnp.linalg.norm(residual)
    scale = jnp.maximum(
        jnp.linalg.norm(capacitance * voltage_mV / dt),
        jnp.finfo(candidate.dtype).tiny,
    )
    relative_residual = residual_norm / scale
    finite = (
        jnp.all(jnp.isfinite(candidate))
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(relative_residual)
    )
    successful = (
        linear_successful
        & finite
        & (relative_residual <= runtime.plan.residual_tolerance)
    )
    return _DiffusionStageResult(
        candidate,
        residual_norm,
        relative_residual,
        linear_status,
        iterations,
        successful,
    )


def _step_evidence(
    runtime: PreparedPhysicalMonodomain,
    state: PhysicalMonodomainState,
    current: Array,
    diffusion_stages: tuple[_DiffusionStageResult, ...],
    reaction_admissible: Array,
    event_aligned: Array,
    candidate_voltage: Array,
    candidate_locals: tuple[Array, ...],
    /,
) -> MonodomainStepEvidence:
    zero_float = jnp.asarray(0.0, dtype=candidate_voltage.dtype)
    zero_int = jnp.asarray(0, dtype=jnp.int32)
    residual_norms = [zero_float, zero_float]
    relative_residuals = [zero_float, zero_float]
    linear_statuses = [zero_int, zero_int]
    iterations = [zero_int, zero_int]
    active = [False, False]
    diffusion_successful = jnp.asarray(True)
    for index, stage in enumerate(diffusion_stages):
        residual_norms[index] = stage.residual_norm_uA
        relative_residuals[index] = stage.relative_residual
        linear_statuses[index] = stage.linear_status
        iterations[index] = stage.iterations
        active[index] = True
        diffusion_successful = diffusion_successful & stage.successful
    residual_array = jnp.stack(residual_norms)
    relative_array = jnp.stack(relative_residuals)
    linear_array = jnp.stack(linear_statuses)
    iteration_array = jnp.stack(iterations)
    active_array = jnp.asarray(active)
    input_finite = jnp.all(jnp.isfinite(current))
    state_finite = jnp.all(jnp.isfinite(candidate_voltage))
    for local in candidate_locals:
        state_finite = state_finite & jnp.all(jnp.isfinite(local))
    finite = (
        input_finite
        & state_finite
        & jnp.all(jnp.isfinite(residual_array))
        & jnp.all(jnp.isfinite(relative_array))
    )
    solve_ok = jnp.all(
        jnp.where(
            active_array,
            linear_array == int(LinearSolveStatus.SUCCESS),
            True,
        )
    )
    residual_ok = jnp.all(
        jnp.where(
            active_array,
            relative_array <= runtime.plan.residual_tolerance,
            True,
        )
    )
    status = jnp.asarray(int(PhysicalMonodomainStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        finite,
        status,
        jnp.bitwise_or(status, int(PhysicalMonodomainStatus.NONFINITE)),
    )
    status = jnp.where(
        solve_ok,
        status,
        jnp.bitwise_or(status, int(PhysicalMonodomainStatus.DIFFUSION_SOLVE_FAILURE)),
    )
    status = jnp.where(
        residual_ok & diffusion_successful,
        status,
        jnp.bitwise_or(status, int(PhysicalMonodomainStatus.DIFFUSION_RESIDUAL_FAILURE)),
    )
    status = jnp.where(
        reaction_admissible,
        status,
        jnp.bitwise_or(status, int(PhysicalMonodomainStatus.REACTION_FAILURE)),
    )
    status = jnp.where(
        input_finite,
        status,
        jnp.bitwise_or(status, int(PhysicalMonodomainStatus.INVALID_INPUT)),
    )
    status = jnp.where(
        event_aligned,
        status,
        jnp.bitwise_or(status, int(PhysicalMonodomainStatus.EVENT_MISALIGNMENT)),
    )
    successful = status == int(PhysicalMonodomainStatus.SUCCESS)
    next_macro = state.macro_step_index + jnp.asarray(1, dtype=jnp.int32)
    checkpoint_due = (next_macro % runtime.plan.schedule.checkpoint_stride) == 0
    calls = jnp.full(
        (len(runtime.worksets),),
        runtime.plan.schedule.reaction_ticks_per_macro,
        dtype=jnp.int32,
    )
    return MonodomainStepEvidence(
        residual_array,
        relative_array,
        linear_array,
        iteration_array,
        active_array,
        reaction_admissible,
        calls,
        calls,
        jnp.asarray(
            [workset.node_indices.shape[0] for workset in runtime.worksets],
            dtype=jnp.int32,
        ),
        event_aligned,
        finite,
        state.tick,
        state.tick + runtime.plan.schedule.reaction_ticks_per_macro,
        jnp.asarray(runtime.plan.schedule.reaction_ticks_per_macro, dtype=jnp.int32),
        jnp.asarray(len(diffusion_stages), dtype=jnp.int32),
        status,
        successful,
        jnp.asarray(False),
        checkpoint_due,
        jnp.asarray(False),
    )


def propose_physical_monodomain_step(
    runtime: PreparedPhysicalMonodomain,
    state: PhysicalMonodomainState,
    inputs: MonodomainMacroInputs,
    /,
) -> PhysicalMonodomainCandidate:
    """Construct one uncommitted Lie or Strang candidate and original evidence."""
    if not isinstance(runtime, PreparedPhysicalMonodomain):
        raise TypeError("runtime must be a PreparedPhysicalMonodomain.")
    state = _require_step_available(runtime, _validate_state(runtime, state))
    current = _macro_inputs(runtime, inputs)
    event_aligned = _event_alignment(runtime, state, current)
    macro_dt = runtime.plan.schedule.macro_dt_ms
    diffusion_stages: tuple[_DiffusionStageResult, ...]
    if isinstance(runtime.plan.splitting, LieSplit):
        reaction_voltage, local_states, admissible = _reaction_advance(
            runtime, state.voltage_mV, state.local_states, current
        )
        diffusion = _diffusion_advance(
            runtime,
            reaction_voltage,
            macro_dt,
            runtime.full_diffusion_solve,
        )
        candidate_voltage = diffusion.voltage_mV
        diffusion_stages = (diffusion,)
    else:
        first = _diffusion_advance(
            runtime,
            state.voltage_mV,
            0.5 * macro_dt,
            runtime.half_diffusion_solve,
        )
        reaction_voltage, local_states, admissible = _reaction_advance(
            runtime, first.voltage_mV, state.local_states, current
        )
        second = _diffusion_advance(
            runtime,
            reaction_voltage,
            0.5 * macro_dt,
            runtime.half_diffusion_solve,
        )
        candidate_voltage = second.voltage_mV
        diffusion_stages = (first, second)
    evidence = _step_evidence(
        runtime,
        state,
        current,
        diffusion_stages,
        admissible,
        event_aligned,
        candidate_voltage,
        local_states,
    )
    proposed_state = PhysicalMonodomainState(
        candidate_voltage,
        local_states,
        state.tick + runtime.plan.schedule.reaction_ticks_per_macro,
        state.macro_step_index + jnp.asarray(1, dtype=jnp.int32),
        current[-1],
        jnp.asarray(True),
        state.checkpoints,
        runtime.runtime_id,
    )
    return PhysicalMonodomainCandidate(state, proposed_state, evidence)


def _write_checkpoint(state: PhysicalMonodomainState, /) -> MonodomainCheckpointBuffer:
    buffer = state.checkpoints
    cursor = buffer.write_cursor
    capacity = buffer.valid.shape[0]
    return MonodomainCheckpointBuffer(
        buffer.voltage_mV.at[cursor].set(state.voltage_mV),
        tuple(
            stored.at[cursor].set(local)
            for stored, local in zip(buffer.local_states, state.local_states, strict=True)
        ),
        buffer.tick.at[cursor].set(state.tick),
        buffer.macro_step_index.at[cursor].set(state.macro_step_index),
        buffer.last_applied_inward_current_uA_per_mm3.at[cursor].set(
            state.last_applied_inward_current_uA_per_mm3
        ),
        buffer.has_previous_stimulus.at[cursor].set(state.has_previous_stimulus),
        buffer.valid.at[cursor].set(True),
        (cursor + 1) % capacity,
        buffer.runtime_id,
    )


def _committed_evidence(
    original: MonodomainStepEvidence,
    checkpoint_written: Array,
    /,
) -> MonodomainStepEvidence:
    return MonodomainStepEvidence(
        original.diffusion_residual_norm_uA,
        original.diffusion_relative_residual,
        original.diffusion_linear_status,
        original.diffusion_iterations,
        original.diffusion_stage_active,
        original.reaction_admissible,
        original.reaction_rate_call_count,
        original.exact_gate_call_count,
        original.workset_node_count,
        original.event_aligned,
        original.finite,
        original.start_tick,
        original.end_tick,
        original.reaction_tick_count,
        original.diffusion_stage_count,
        original.status,
        original.successful,
        ~original.successful,
        original.checkpoint_due,
        checkpoint_written,
    )


def commit_physical_monodomain_candidate(
    runtime: PreparedPhysicalMonodomain,
    state: PhysicalMonodomainState,
    candidate: PhysicalMonodomainCandidate,
    /,
) -> PhysicalMonodomainStepResult:
    """Commit successful evidence or atomically retain the complete prior state."""
    state = _validate_state(runtime, state)
    if not isinstance(candidate, PhysicalMonodomainCandidate):
        raise TypeError("candidate must be a PhysicalMonodomainCandidate.")
    source_state = _validate_state(runtime, candidate.source_state)
    candidate_state = _validate_state(runtime, candidate.state)
    source_matches = _states_exactly_equal(source_state, state)
    checked_voltage = eqx.error_if(
        state.voltage_mV,
        ~source_matches,
        "PhysicalMonodomainCandidate source state does not match commit state.",
    )
    state = eqx.tree_at(lambda value: value.voltage_mV, state, checked_voltage)
    accepted = jax.tree.map(
        lambda proposed, prior: jnp.where(candidate.evidence.successful, proposed, prior),
        candidate_state,
        state,
    )
    write_checkpoint = candidate.evidence.successful & candidate.evidence.checkpoint_due
    written_buffer = _write_checkpoint(accepted)
    checkpoint_buffer = jax.tree.map(
        lambda written, existing: jnp.where(write_checkpoint, written, existing),
        written_buffer,
        accepted.checkpoints,
    )
    accepted = PhysicalMonodomainState(
        accepted.voltage_mV,
        accepted.local_states,
        accepted.tick,
        accepted.macro_step_index,
        accepted.last_applied_inward_current_uA_per_mm3,
        accepted.has_previous_stimulus,
        checkpoint_buffer,
        runtime.runtime_id,
    )
    evidence = _committed_evidence(candidate.evidence, write_checkpoint)
    return PhysicalMonodomainStepResult(accepted, candidate, evidence)


def step_physical_monodomain(
    runtime: PreparedPhysicalMonodomain,
    state: PhysicalMonodomainState,
    inputs: MonodomainMacroInputs,
    /,
) -> PhysicalMonodomainStepResult:
    """Propose, check, and fail-closed commit one macro step."""
    candidate = propose_physical_monodomain_step(runtime, state, inputs)
    return commit_physical_monodomain_candidate(runtime, state, candidate)


def integrate_physical_monodomain(
    runtime: PreparedPhysicalMonodomain,
    state: PhysicalMonodomainState,
    inputs: ScheduledMonodomainInputs,
    /,
) -> MonodomainIntegrationResult:
    """Execute the complete fixed integer schedule with immutable evidence."""
    state = _validate_state(runtime, state)
    checked_voltage = eqx.error_if(
        state.voltage_mV,
        (state.tick != 0) | (state.macro_step_index != 0),
        "Fixed-horizon integration requires the initialized schedule state.",
    )
    state = eqx.tree_at(lambda value: value.voltage_mV, state, checked_voltage)
    if not isinstance(inputs, ScheduledMonodomainInputs):
        raise TypeError("inputs must be ScheduledMonodomainInputs.")
    current = jnp.asarray(inputs.inward_current_uA_per_mm3)
    expected = (
        runtime.plan.schedule.total_tick_count,
        runtime.plan.node_count,
    )
    if current.shape != expected:
        raise ValueError(f"inward_current_uA_per_mm3 must have shape {expected}.")
    if current.dtype != runtime.spatial.node_volume_mm3.dtype:
        raise TypeError("Scheduled inputs must use the prepared runtime dtype.")
    stride = runtime.plan.schedule.reaction_ticks_per_macro

    def body(current_state, macro_index):
        start = macro_index * stride
        macro_current = jax.lax.dynamic_slice(
            current,
            (start, jnp.asarray(0, dtype=macro_index.dtype)),
            (stride, runtime.plan.node_count),
        )
        result = step_physical_monodomain(
            runtime, current_state, MonodomainMacroInputs(macro_current)
        )
        return result.state, (result.candidate.state.voltage_mV, result.evidence)

    final_state, (candidate_voltage, evidence) = jax.lax.scan(
        body,
        state,
        jnp.arange(runtime.plan.schedule.macro_step_count, dtype=jnp.int32),
    )
    return MonodomainIntegrationResult(final_state, candidate_voltage, evidence)


def rollback_physical_monodomain(
    runtime: PreparedPhysicalMonodomain,
    state: PhysicalMonodomainState,
    /,
) -> MonodomainRollbackResult:
    """Restore the newest valid complete checkpoint without trusting live dynamics."""
    state = _validate_state_structure(runtime, state)
    buffer = state.checkpoints
    capacity = buffer.valid.shape[0]
    stride = runtime.plan.schedule.reaction_ticks_per_macro
    checkpoint_valid = (
        buffer.valid
        & (buffer.tick >= 0)
        & (buffer.macro_step_index >= 0)
        & (buffer.macro_step_index <= runtime.plan.schedule.macro_step_count)
        & (buffer.tick <= runtime.plan.schedule.total_tick_count)
        & (buffer.tick == buffer.macro_step_index * stride)
        & jnp.all(jnp.isfinite(buffer.voltage_mV), axis=1)
        & jnp.all(
            jnp.isfinite(buffer.last_applied_inward_current_uA_per_mm3),
            axis=1,
        )
    )
    for stored, workset in zip(buffer.local_states, runtime.worksets, strict=True):
        local_finite = jnp.all(jnp.isfinite(stored), axis=tuple(range(1, stored.ndim)))

        def admissible(voltage, local):
            return jnp.all(
                workset.reaction.admissible(voltage[workset.node_indices], local)
            )

        local_admissible = jax.vmap(admissible)(buffer.voltage_mV, stored)
        checkpoint_valid = checkpoint_valid & local_finite & local_admissible
    newest_first = (
        buffer.write_cursor - 1 - jnp.arange(capacity, dtype=buffer.write_cursor.dtype)
    ) % capacity
    ordered_valid = checkpoint_valid[newest_first]
    found = jnp.any(ordered_valid)
    selected = newest_first[jnp.argmax(ordered_valid.astype(jnp.int32))]
    evidence_slot = jnp.where(found, selected, -1)
    evidence_tick = jnp.where(found, buffer.tick[selected], -1)
    restored = PhysicalMonodomainState(
        jnp.where(found, buffer.voltage_mV[selected], state.voltage_mV),
        tuple(
            jnp.where(found, stored[selected], local)
            for stored, local in zip(buffer.local_states, state.local_states, strict=True)
        ),
        jnp.where(found, buffer.tick[selected], state.tick),
        jnp.where(found, buffer.macro_step_index[selected], state.macro_step_index),
        jnp.where(
            found,
            buffer.last_applied_inward_current_uA_per_mm3[selected],
            state.last_applied_inward_current_uA_per_mm3,
        ),
        jnp.where(
            found,
            buffer.has_previous_stimulus[selected],
            state.has_previous_stimulus,
        ),
        buffer,
        runtime.runtime_id,
    )
    evidence = MonodomainRollbackEvidence(evidence_slot, evidence_tick, found)
    return MonodomainRollbackResult(restored, evidence)


__all__ = [
    "DiffusionOperatorInput",
    "EventAlignedMultirateSchedule",
    "ExplicitReferenceDiffusion",
    "HomogeneousReactionWorkset",
    "ImplicitThetaDiffusion",
    "LieSplit",
    "MonodomainCheckpointBuffer",
    "MonodomainDiffusionMethod",
    "MonodomainIntegrationResult",
    "MonodomainMacroInputs",
    "MonodomainRollbackEvidence",
    "MonodomainRollbackResult",
    "MonodomainSplitting",
    "PhysicalMonodomainStatus",
    "MonodomainStepEvidence",
    "PhysicalMonodomainCandidate",
    "PhysicalMonodomainPlan",
    "PhysicalMonodomainSpatialBinding",
    "PhysicalMonodomainState",
    "PhysicalMonodomainStepResult",
    "PreparedPhysicalMonodomain",
    "PublicDiffusionOperatorInput",
    "ScheduledMonodomainInputs",
    "StrangSplit",
    "TensorDiffusionOperatorInput",
    "commit_physical_monodomain_candidate",
    "initialize_physical_monodomain",
    "integrate_physical_monodomain",
    "prepare_physical_monodomain",
    "propose_physical_monodomain_step",
    "rollback_physical_monodomain",
    "step_physical_monodomain",
    "zero_monodomain_macro_inputs",
    "zero_scheduled_monodomain_inputs",
]
