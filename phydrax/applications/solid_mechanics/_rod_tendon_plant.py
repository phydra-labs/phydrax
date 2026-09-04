#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import prod
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._array_tree import ArrayPyTreeSchema
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._identity import ExecutableSignature, NumericRevision, SemanticProvenance
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import (
    AbstractDiscretePlant,
    PlantParameters,
    PlantProposal,
    PlantRuntimeState,
    PlantStepContext,
)
from ...dynamics._layout import StateLayout
from ...dynamics._plant_codec import ControlVectorCodec, PlantStateVectorCodec
from ...linalg import ArraySpace, BlockSpace
from ...nonlinear import NonlinearSystemProblem
from ._rod_plant import (
    PreparedReducedRodPlant,
    ReducedRodMassResponseRevision,
    ReducedRodPassiveContactState,
    ReducedRodPassiveSensorState,
    ReducedRodPlantState,
)
from ._rod_reduced_dynamics import ReducedRodDirectLoad, ReducedRodMaterialState
from ._rod_reduced_integrators import (
    _candidate_state,
    _energy_work_ledger,
    _selected_state,
    _source_mechanics_valid,
    _status,
    ReducedRodImplicitMidpoint,
    ReducedRodIntegrationState,
    ReducedRodSemiImplicitVelocityEuler,
    ReducedRodStepEvidence,
    ReducedRodStepResult,
)
from ._rod_reduction import ReducedRodState
from ._rod_tendon import (
    PreparedFrictionlessElasticTendon,
    TendonActuationEvaluation,
    TendonActuatorState,
    TendonPayoutCommand,
)


if TYPE_CHECKING:
    from ..robotics._soft_observations import (
        PreparedSoftObservationPlan,
        SoftRobotObservation,
        SoftSensorState,
    )


class TendonDrivenRodPlantStatus(IntEnum):
    SUCCESS = 0
    STEP_OUT_OF_BOUNDS = 1
    SOURCE_INVALID = 2
    MASS_SOLVE_FAILED = 3
    NONLINEAR_SOLVE_FAILED = 4
    MATERIAL_TRIAL_FAILED = 5
    CANDIDATE_INVALID = 6
    MECHANICAL_LEDGER_INVALID = 7
    COMMAND_OUT_OF_BOUNDS = 8
    TENDON_SOURCE_INVALID = 9
    TENDON_CANDIDATE_INVALID = 10
    TENDON_LEDGER_INVALID = 11
    SENSOR_INVALID = 12


class TendonActuatorStateBank(StrictModule):
    states: tuple[TendonActuatorState, ...]

    def __init__(self, states: Sequence[TendonActuatorState], /):
        values = tuple(states)
        if not values or any(
            not isinstance(value, TendonActuatorState) for value in values
        ):
            raise TypeError("states must contain one or more TendonActuatorState values.")
        self.states = values


class TendonDrivenRodPlantState(StrictModule):
    """Complete mechanics, actuator, contact, and sensor transaction payload."""

    reduced_state: ReducedRodState
    material_state: ReducedRodMaterialState
    actuator_state: TendonActuatorStateBank
    contact_state: ReducedRodPassiveContactState
    sensor_state: ReducedRodPassiveSensorState | SoftSensorState


class TendonDrivenRodPlantParameters(StrictModule):
    values: Array

    def __init__(self, values: ArrayLike, /):
        result = jnp.asarray(values)
        if result.shape != (0,):
            raise ValueError("Tendon plant parameters must have shape (0,).")
        if not jnp.issubdtype(result.dtype, jnp.inexact) or jnp.iscomplexobj(result):
            raise TypeError("Tendon plant parameters must use a real inexact dtype.")
        self.values = result


class TendonDrivenRodPlantCommand(StrictModule):
    """Nonempty tendon command bank plus a true reduced external effort."""

    tendon_commands: tuple[TendonPayoutCommand, ...]
    external_effort: Array

    def __init__(
        self,
        tendon_commands: Sequence[TendonPayoutCommand],
        external_effort: ArrayLike,
        /,
    ):
        commands = tuple(tendon_commands)
        if not commands or any(
            not isinstance(value, TendonPayoutCommand) for value in commands
        ):
            raise TypeError("tendon_commands must be a nonempty tendon command sequence.")
        effort = jnp.asarray(external_effort)
        if effort.ndim != 1:
            raise ValueError("external_effort must be rank one.")
        if not jnp.issubdtype(effort.dtype, jnp.inexact) or jnp.iscomplexobj(effort):
            raise TypeError("external_effort must be a real inexact array.")
        self.tendon_commands = commands
        self.external_effort = effort


class TendonDrivenRodCommandBounds(StrictModule, NonTrainableState):
    lower: TendonDrivenRodPlantCommand
    upper: TendonDrivenRodPlantCommand
    bounds_id: str = eqx.field(static=True)

    def contains(self, command: TendonDrivenRodPlantCommand, /) -> Array:
        rates = jnp.stack(tuple(value.payout_rate for value in command.tendon_commands))
        lower = jnp.stack(
            tuple(value.payout_rate for value in self.lower.tendon_commands)
        )
        upper = jnp.stack(
            tuple(value.payout_rate for value in self.upper.tendon_commands)
        )
        return (
            jnp.all(rates >= lower)
            & jnp.all(rates <= upper)
            & jnp.all(command.external_effort >= self.lower.external_effort)
            & jnp.all(command.external_effort <= self.upper.external_effort)
        )


class TendonDrivenRodActuationLedger(StrictModule):
    payout_evaluations: tuple[TendonActuationEvaluation, ...]
    source_evaluations: tuple[TendonActuationEvaluation, ...]
    candidate_evaluations: tuple[TendonActuationEvaluation, ...]
    source_tension: Array
    candidate_tension: Array
    source_stored_energy: Array
    candidate_stored_energy: Array
    spool_work: Array
    rod_work: Array
    energy_residual: Array
    total_spool_work: Array
    total_rod_work: Array
    total_energy_residual: Array
    finite: Array
    source_valid: Array
    candidate_valid: Array
    balanced: Array
    valid: Array
    tendon_ids: tuple[str, ...] = eqx.field(static=True)


class TendonDrivenRodPlantResetEvidence(StrictModule):
    tendon_evaluations: tuple[TendonActuationEvaluation, ...]
    observation: SoftRobotObservation | None
    observation_valid: Array
    finite: Array
    valid: Array
    status: Array
    plant_id: str = eqx.field(static=True)


class TendonDrivenRodPlantEvidence(StrictModule):
    integration_result: ReducedRodStepResult
    tendon_ledger: TendonDrivenRodActuationLedger
    observation: SoftRobotObservation | None
    command_within_bounds: Array
    observation_valid: Array
    finite: Array
    valid: Array
    status: Array
    plant_id: str = eqx.field(static=True)
    tendon_ids: tuple[str, ...] = eqx.field(static=True)


class TendonDrivenRodMassResponseRevision(StrictModule):
    base_response: ReducedRodMassResponseRevision
    time: Array
    step_index: Array
    plant_id: str = eqx.field(static=True)
    semantic_provenance_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    state_schema_id: str = eqx.field(static=True)
    execution_signature_id: str = eqx.field(static=True)

    @property
    def finite(self) -> Array:
        return self.base_response.finite

    @property
    def valid(self) -> Array:
        return self.base_response.valid

    def apply_impulse(self, impulse: ArrayLike, /) -> Array:
        return self.base_response.apply_impulse(impulse)

    def is_current(self, state: PlantRuntimeState, /) -> Array:
        if not isinstance(state.payload, TendonDrivenRodPlantState):
            raise TypeError("Runtime payload must be TendonDrivenRodPlantState.")
        if (
            state.semantic_provenance_id,
            state.numeric_revision_id,
            state.state_schema_id,
            state.execution_signature_id,
        ) != (
            self.semantic_provenance_id,
            self.numeric_revision_id,
            self.state_schema_id,
            self.execution_signature_id,
        ):
            raise ValueError("Mass response and runtime identities differ.")
        return (
            (state.time == self.time)
            & (state.step_index == self.step_index)
            & jnp.all(
                state.payload.reduced_state.coefficients
                == self.base_response.configuration
            )
            & jnp.all(
                state.payload.reduced_state.coefficient_velocities
                == self.base_response.free_velocity
            )
        )


def _all(values: Sequence[Array], /) -> Array:
    result = jnp.asarray(True)
    for value in values:
        result = result & value
    return result


class _TendonMidpointResidual(StrictModule):
    dynamics: Any
    tendons: tuple[PreparedFrictionlessElasticTendon, ...]
    actuator_state: TendonActuatorStateBank
    external_effort: Array
    source: ReducedRodIntegrationState
    material_control: Any
    native_loads: Any
    step_size: Array
    tendon_ids: tuple[str, ...] = eqx.field(static=True)

    def __call__(self, state: tuple[Array, Array], _arguments: Any, /):
        q0 = self.source.reduced_state.coefficients
        v0 = self.source.reduced_state.coefficient_velocities
        q1, v1 = state
        midpoint = ReducedRodState(0.5 * (q0 + q1), 0.5 * (v0 + v1))
        evaluations = tuple(
            tendon.evaluate(
                midpoint,
                actuator,
                TendonPayoutCommand(jnp.zeros_like(actuator.free_length)),
            )
            for tendon, actuator in zip(
                self.tendons, self.actuator_state.states, strict=True
            )
        )
        direct_loads = tuple(
            ReducedRodDirectLoad(
                evaluation.reduced_effort,
                source_id=f"tendon:{tendon_id}",
                power_channel="tendon",
            )
            for evaluation, tendon_id in zip(evaluations, self.tendon_ids, strict=True)
        ) + (
            ReducedRodDirectLoad(
                self.external_effort,
                source_id="external-reduced-command",
                power_channel="external-command",
            ),
        )
        inverse = self.dynamics.inverse_dynamics(
            midpoint,
            (v1 - v0) / self.step_size,
            source_state=self.source.reduced_state,
            material_state=self.source.material_state,
            material_control=self.material_control,
            time=self.source.time + 0.5 * self.step_size,
            step_size=0.5 * self.step_size,
            native_loads=self.native_loads,
            direct_reduced_loads=direct_loads,
        )
        kinematic = q1 - q0 - 0.5 * self.step_size * (v0 + v1)
        return (kinematic, inverse.residual), inverse


class PreparedTendonDrivenRodPlant(AbstractDiscretePlant, NonTrainableState):
    """Contact-free reduced rod with atomic payout and mechanical acceptance."""

    base_plant: PreparedReducedRodPlant
    tendons: tuple[PreparedFrictionlessElasticTendon, ...]
    initial_state: TendonDrivenRodPlantState
    default_parameters: TendonDrivenRodPlantParameters
    command_bounds: TendonDrivenRodCommandBounds
    observation_plan: PreparedSoftObservationPlan | None
    state_schema: ArrayPyTreeSchema
    control_schema: ArrayPyTreeSchema
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: TendonDrivenRodPlantState
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    execution_signature: ExecutableSignature
    state_codec: PlantStateVectorCodec
    control_codec: ControlVectorCodec
    require_finite_state: bool = eqx.field(static=True)
    require_finite_controls: bool = eqx.field(static=True)
    require_finite_parameters: bool = eqx.field(static=True)
    tendon_ids: tuple[str, ...] = eqx.field(static=True)
    plant_id: str = eqx.field(static=True)

    def __init__(
        self,
        base_plant: PreparedReducedRodPlant,
        tendons: Sequence[PreparedFrictionlessElasticTendon],
        initial_free_lengths: Sequence[ArrayLike],
        /,
        *,
        external_effort_bounds: tuple[ArrayLike, ArrayLike] | None = None,
        observation_plan: PreparedSoftObservationPlan | None = None,
        initial_sensor_state: SoftSensorState | None = None,
    ):
        from ..robotics._soft_observations import (
            PreparedSoftObservationPlan,
        )

        if not isinstance(base_plant, PreparedReducedRodPlant):
            raise TypeError("base_plant must be PreparedReducedRodPlant.")
        tendon_values = tuple(tendons)
        lengths = tuple(initial_free_lengths)
        if not tendon_values or any(
            not isinstance(value, PreparedFrictionlessElasticTendon)
            for value in tendon_values
        ):
            raise TypeError("tendons must be a nonempty prepared tendon sequence.")
        if len(lengths) != len(tendon_values):
            raise ValueError("initial_free_lengths must contain one value per tendon.")
        if not isinstance(
            base_plant.policy,
            (ReducedRodSemiImplicitVelocityEuler, ReducedRodImplicitMidpoint),
        ):
            raise TypeError("The tendon plant requires one reduced integrator route.")
        reduction_id = base_plant.dynamics.reduction.prepared_id
        if any(
            tendon.route.reduction is None
            or tendon.route.reduction.prepared_id != reduction_id
            for tendon in tendon_values
        ):
            raise ValueError("Every tendon must use the base plant reduction.")
        tendon_ids = tuple(tendon.tendon_id for tendon in tendon_values)
        if len(set(tendon_ids)) != len(tendon_ids):
            raise ValueError("Every tendon_id must be unique.")
        dtype = base_plant.initial_state.reduced_state.values.dtype
        actuator = TendonActuatorStateBank(
            tuple(
                tendon.initialize_state(jnp.asarray(length, dtype=dtype))
                for tendon, length in zip(tendon_values, lengths, strict=True)
            )
        )
        empty = jnp.zeros((0,), dtype=dtype)
        if observation_plan is None:
            if initial_sensor_state is not None:
                raise ValueError("initial_sensor_state requires observation_plan.")
            sensor: ReducedRodPassiveSensorState | SoftSensorState = (
                ReducedRodPassiveSensorState(empty)
            )
        else:
            if not isinstance(observation_plan, PreparedSoftObservationPlan):
                raise TypeError("observation_plan has the wrong type.")
            if observation_plan.plant.plant_id != base_plant.plant_id:
                raise ValueError("observation_plan must be prepared against base_plant.")
            if (
                observation_plan.energy_load is not None
                and observation_plan.energy_load.plan.include_step_ledger
            ):
                raise ValueError(
                    "Tendon plants cannot expose a base-plant step ledger observation."
                )
            if (
                observation_plan.tendon is not None
                and observation_plan.tendon.plan.requires_actuator_state
                and len(observation_plan.tendon.plan.tendons) != len(tendon_values)
            ):
                raise ValueError(
                    "Stateful tendon observations require one query tendon per plant tendon."
                )
            if observation_plan.sensor is None:
                sensor = ReducedRodPassiveSensorState(empty)
            else:
                sensor = (
                    observation_plan.initialize_sensor_state()
                    if initial_sensor_state is None
                    else initial_sensor_state
                )
                observation_plan.sensor._validate_state(sensor)
        initial = TendonDrivenRodPlantState(
            base_plant.initial_state.reduced_state,
            base_plant.initial_state.material_state,
            actuator,
            ReducedRodPassiveContactState(empty),
            sensor,
        )
        coordinate_count = base_plant.dynamics.reduction.plan.coordinate_count
        zero = TendonDrivenRodPlantCommand(
            tuple(
                TendonPayoutCommand(jnp.asarray(0.0, dtype=dtype)) for _ in tendon_values
            ),
            jnp.zeros((coordinate_count,), dtype=dtype),
        )
        state_schema = ArrayPyTreeSchema.from_tree(initial, case_ndim=0)
        control_schema = ArrayPyTreeSchema.from_tree(zero, case_ndim=0)
        parameters = TendonDrivenRodPlantParameters(empty)
        parameter_schema = ArrayPyTreeSchema.from_tree(parameters, case_ndim=0)
        if external_effort_bounds is None:
            magnitude = jnp.asarray(jnp.finfo(dtype).max, dtype=dtype)
            effort_lower = jnp.full((coordinate_count,), -magnitude, dtype=dtype)
            effort_upper = jnp.full((coordinate_count,), magnitude, dtype=dtype)
        else:
            effort_lower = jnp.asarray(external_effort_bounds[0], dtype=dtype)
            effort_upper = jnp.asarray(external_effort_bounds[1], dtype=dtype)
            if effort_lower.shape != (coordinate_count,) or effort_upper.shape != (
                coordinate_count,
            ):
                raise ValueError("External effort bounds must match coordinate_count.")
        lower = TendonDrivenRodPlantCommand(
            tuple(
                TendonPayoutCommand(
                    jnp.asarray(value.plan.minimum_payout_rate, dtype=dtype)
                )
                for value in tendon_values
            ),
            effort_lower,
        )
        upper = TendonDrivenRodPlantCommand(
            tuple(
                TendonPayoutCommand(
                    jnp.asarray(value.plan.maximum_payout_rate, dtype=dtype)
                )
                for value in tendon_values
            ),
            effort_upper,
        )
        bounds = TendonDrivenRodCommandBounds(
            lower,
            upper,
            canonical_fingerprint(
                {
                    "kind": "tendon-command-bounds",
                    "lower": array_tree_fingerprint(lower),
                    "upper": array_tree_fingerprint(upper),
                }
            ),
        )
        semantic = SemanticProvenance(
            {
                "kind": "contact-free-tendon-driven-reduced-rod-plant",
                "base": base_plant.semantic_provenance.semantic_id,
                "reduction": reduction_id,
                "tendons": tendon_ids,
                "state_schema": state_schema.content_id,
                "control_schema": control_schema.content_id,
                "observation": (
                    None
                    if observation_plan is None
                    else observation_plan.observation_plan_id
                ),
                "contact": "none",
            }
        )
        numeric = NumericRevision(
            semantic,
            {
                "base": base_plant.numeric_revision.revision_id,
                "initial": initial,
                "bounds": bounds,
            },
        )
        signature = ExecutableSignature(
            shapes=tuple(
                (f"state:{leaf.path}", leaf.shape) for leaf in state_schema.leaves
            )
            + tuple(
                (f"control:{leaf.path}", leaf.shape) for leaf in control_schema.leaves
            ),
            dtypes=tuple(
                (f"state:{leaf.path}", leaf.dtype) for leaf in state_schema.leaves
            )
            + tuple(
                (f"control:{leaf.path}", leaf.dtype) for leaf in control_schema.leaves
            ),
            space_ids={
                "configuration": base_plant.dynamics.reduction.coefficient_space.space_id,
                "effort": base_plant.dynamics.reduction.reduced_effort_space.space_id,
            },
            topology_ids={
                "native_rod": base_plant.dynamics.reduction.rod.prepared_id,
                "reduction": reduction_id,
                **{f"tendon_{index}": value for index, value in enumerate(tendon_ids)},
            },
            capacities={"coordinates": coordinate_count, "tendons": len(tendon_values)},
            algorithm_facts={"integrator": base_plant.policy.route, "contact": "none"},
        )
        immutable_mode_paths: tuple[str, ...] = ()
        dynamic_size = sum(
            prod(leaf.shape) for leaf in state_schema.leaves if leaf.dtype.kind in "fc"
        )
        state_codec = PlantStateVectorCodec(
            state_schema,
            StateLayout(
                (dynamic_size,),
                axes=("coordinate",),
                local_space=ArraySpace((dynamic_size,), dtype=dtype),
                tangent_space=ArraySpace((dynamic_size,), dtype=dtype),
            ),
            initial,
            immutable_mode_paths,
            semantic_provenance=semantic,
            numeric_revision=numeric,
            executable_signature=signature,
        )
        plant_id = canonical_fingerprint(
            {
                "kind": "prepared-tendon-driven-rod-plant",
                "numeric": numeric.revision_id,
                "execution": signature.signature_id,
            }
        )
        self.base_plant = base_plant
        self.tendons = tendon_values
        self.initial_state = initial
        self.default_parameters = parameters
        self.command_bounds = bounds
        self.observation_plan = observation_plan
        self.state_schema = state_schema
        self.control_schema = control_schema
        self.parameter_schema = parameter_schema
        self.reset_fallback = initial
        self.semantic_provenance = semantic
        self.numeric_revision = numeric
        self.execution_signature = signature
        self.state_codec = state_codec
        self.control_codec = ControlVectorCodec(
            control_schema,
            semantic_provenance=semantic,
            numeric_revision=numeric,
            executable_signature=signature,
        )
        self.require_finite_state = True
        self.require_finite_controls = True
        self.require_finite_parameters = True
        self.tendon_ids = tendon_ids
        self.plant_id = plant_id

    @property
    def dynamics(self):
        return self.base_plant.dynamics

    def bind_parameters(
        self, values: TendonDrivenRodPlantParameters | None = None, /
    ) -> PlantParameters:
        resolved = self.default_parameters if values is None else values
        self.parameter_schema.validate(resolved)
        return PlantParameters(
            resolved, self.parameter_schema.schema_id, self.numeric_revision
        )

    def command(
        self,
        payout_rates: Sequence[ArrayLike],
        /,
        *,
        external_effort: ArrayLike | None = None,
    ) -> TendonDrivenRodPlantCommand:
        if len(tuple(payout_rates)) != len(self.tendons):
            raise ValueError("payout_rates must contain one value per tendon.")
        dtype = self.initial_state.reduced_state.values.dtype
        effort = (
            jnp.zeros((self.dynamics.reduction.plan.coordinate_count,), dtype=dtype)
            if external_effort is None
            else jnp.asarray(external_effort, dtype=dtype)
        )
        result = TendonDrivenRodPlantCommand(
            tuple(
                TendonPayoutCommand(jnp.asarray(value, dtype=dtype))
                for value in payout_rates
            ),
            effort,
        )
        self.control_schema.validate(result)
        return result

    def zero_command(self) -> TendonDrivenRodPlantCommand:
        return self.command((0.0,) * len(self.tendons))

    def _loaded_evaluations(
        self, rod_state: ReducedRodState, actuator: TendonActuatorStateBank, /
    ) -> tuple[TendonActuationEvaluation, ...]:
        return tuple(
            tendon.evaluate(
                rod_state,
                state,
                TendonPayoutCommand(jnp.zeros_like(state.free_length)),
            )
            for tendon, state in zip(self.tendons, actuator.states, strict=True)
        )

    def _loads(
        self,
        evaluations: tuple[TendonActuationEvaluation, ...],
        external_effort: Array,
        /,
    ) -> tuple[ReducedRodDirectLoad, ...]:
        return tuple(
            ReducedRodDirectLoad(
                value.reduced_effort,
                source_id=f"tendon:{tendon_id}",
                power_channel="tendon",
            )
            for value, tendon_id in zip(evaluations, self.tendon_ids, strict=True)
        ) + (
            ReducedRodDirectLoad(
                external_effort,
                source_id="external-reduced-command",
                power_channel="external-command",
            ),
        )

    def _payout(
        self,
        source: TendonDrivenRodPlantState,
        commands: tuple[TendonPayoutCommand, ...],
        step: Array,
        /,
    ) -> tuple[TendonActuatorStateBank, tuple[TendonActuationEvaluation, ...]]:
        evaluations = tuple(
            tendon.evaluate(
                source.reduced_state,
                state,
                command,
                time_step=step,
            )
            for tendon, state, command in zip(
                self.tendons,
                source.actuator_state.states,
                commands,
                strict=True,
            )
        )
        return (
            TendonActuatorStateBank(
                tuple(evaluation.candidate_state for evaluation in evaluations)
            ),
            evaluations,
        )

    def _semi_implicit_integration(
        self,
        source: TendonDrivenRodPlantState,
        actuator: TendonActuatorStateBank,
        command: TendonDrivenRodPlantCommand,
        context: PlantStepContext,
        /,
    ) -> ReducedRodStepResult:
        policy = self.base_plant.policy
        if not isinstance(policy, ReducedRodSemiImplicitVelocityEuler):
            raise TypeError("The prepared plant does not use semi-implicit integration.")
        integration_source = ReducedRodIntegrationState(
            source.reduced_state,
            source.material_state,
            context.source_time,
            context.step_index,
        )
        source_tendons = self._loaded_evaluations(source.reduced_state, actuator)
        forward = self.dynamics.forward_dynamics(
            source.reduced_state,
            material_state=source.material_state,
            material_control=self.base_plant.material_control,
            time=context.source_time,
            step_size=context.duration,
            native_loads=self.base_plant.native_loads,
            direct_reduced_loads=self._loads(source_tendons, command.external_effort),
        )
        velocity = (
            source.reduced_state.coefficient_velocities
            + context.duration * forward.acceleration
        )
        candidate_reduced = ReducedRodState(
            source.reduced_state.coefficients + context.duration * velocity,
            velocity,
        )
        candidate_tendons = self._loaded_evaluations(candidate_reduced, actuator)
        candidate_evaluation = self.dynamics.evaluate(
            candidate_reduced,
            source_state=source.reduced_state,
            material_state=source.material_state,
            material_control=self.base_plant.material_control,
            time=context.target_time,
            step_size=context.duration,
            native_loads=self.base_plant.native_loads,
            direct_reduced_loads=self._loads(candidate_tendons, command.external_effort),
        )
        candidate = _candidate_state(
            integration_source,
            candidate_reduced,
            candidate_evaluation.candidate_material_state,
            context.duration,
        )
        ledger = _energy_work_ledger(
            forward.evaluation,
            candidate_evaluation,
            context.duration,
            policy.energy_balance_tolerance,
        )
        step_finite = jnp.isfinite(context.duration) & (context.duration > 0.0)
        step_valid = step_finite & (context.duration <= policy.maximum_step_size)
        source_valid = _source_mechanics_valid(forward.evaluation)
        linear_valid = forward.solve_evidence.successful
        material_valid = (
            candidate_evaluation.stretch_shear_material_result.evidence.valid
            & candidate_evaluation.bend_twist_material_result.evidence.valid
        )
        candidate_valid = candidate_evaluation.valid
        finite = forward.finite & candidate_evaluation.finite & ledger.finite
        successful = (
            step_valid
            & source_valid
            & linear_valid
            & material_valid
            & candidate_valid
            & ledger.valid
            & finite
        )
        status = _status(
            step_valid=step_valid,
            source_valid=source_valid,
            linear_valid=linear_valid,
            nonlinear_attempted=False,
            nonlinear_valid=jnp.asarray(True),
            material_valid=material_valid,
            candidate_valid=candidate_valid,
            ledger_valid=ledger.valid & finite,
        )
        backend = jnp.asarray(forward.solve_evidence.status, dtype=jnp.int32)
        evidence = ReducedRodStepEvidence(
            forward.evaluation,
            candidate_evaluation,
            forward.solve_evidence,
            None,
            ledger,
            step_finite,
            step_valid,
            source_valid,
            linear_valid,
            jnp.asarray(True),
            material_valid,
            candidate_valid,
            finite,
            successful,
            status,
            backend,
            policy.route,
            policy.policy_id,
        )
        return ReducedRodStepResult(
            integration_source,
            candidate,
            _selected_state(successful, candidate, integration_source),
            jnp.asarray(True),
            successful,
            status,
            backend,
            evidence,
            policy.policy_id,
        )

    def _implicit_midpoint_integration(
        self,
        source: TendonDrivenRodPlantState,
        actuator: TendonActuatorStateBank,
        command: TendonDrivenRodPlantCommand,
        context: PlantStepContext,
        /,
    ) -> ReducedRodStepResult:
        policy = self.base_plant.policy
        if not isinstance(policy, ReducedRodImplicitMidpoint):
            raise TypeError(
                "The prepared plant does not use implicit-midpoint integration."
            )
        integration_source = ReducedRodIntegrationState(
            source.reduced_state,
            source.material_state,
            context.source_time,
            context.step_index,
        )
        source_tendons = self._loaded_evaluations(source.reduced_state, actuator)
        source_evaluation = self.dynamics.evaluate(
            source.reduced_state,
            material_state=source.material_state,
            material_control=self.base_plant.material_control,
            time=context.source_time,
            step_size=context.duration,
            native_loads=self.base_plant.native_loads,
            direct_reduced_loads=self._loads(source_tendons, command.external_effort),
        )
        q0 = source.reduced_state.coefficients
        v0 = source.reduced_state.coefficient_velocities
        initial = (q0 + context.duration * v0, v0)
        state_space = BlockSpace(
            (
                self.dynamics.reduction.coefficient_space,
                self.dynamics.reduction.coefficient_space,
            ),
            names=("configuration", "velocity"),
        )
        residual_space = BlockSpace(
            (
                self.dynamics.reduction.coefficient_space,
                self.dynamics.reduction.reduced_effort_space,
            ),
            names=("kinematic", "dynamic"),
        )
        residual = _TendonMidpointResidual(
            self.dynamics,
            self.tendons,
            actuator,
            command.external_effort,
            integration_source,
            self.base_plant.material_control,
            self.base_plant.native_loads,
            context.duration,
            self.tendon_ids,
        )
        problem = NonlinearSystemProblem(
            residual,
            state_space=state_space,
            residual_space=residual_space,
            has_aux=True,
            validity=lambda _state, _value, inverse, _args: inverse.valid,
            problem_id=(
                f"tendon-reduced-rod-implicit-midpoint:"
                f"{self.dynamics.dynamics_id}:{policy.policy_id}:{self.plant_id}"
            ),
        )
        nonlinear = policy.nonlinear_method.solve(
            problem,
            initial,
            termination=policy.nonlinear_termination,
        )
        q1, v1 = state_space.validate(nonlinear.state)
        candidate_reduced = ReducedRodState(q1, v1)
        candidate_tendons = self._loaded_evaluations(candidate_reduced, actuator)
        candidate_evaluation = self.dynamics.evaluate(
            candidate_reduced,
            source_state=source.reduced_state,
            material_state=source.material_state,
            material_control=self.base_plant.material_control,
            time=context.target_time,
            step_size=context.duration,
            native_loads=self.base_plant.native_loads,
            direct_reduced_loads=self._loads(candidate_tendons, command.external_effort),
        )
        candidate = _candidate_state(
            integration_source,
            candidate_reduced,
            candidate_evaluation.candidate_material_state,
            context.duration,
        )
        ledger = _energy_work_ledger(
            source_evaluation,
            candidate_evaluation,
            context.duration,
            policy.energy_balance_tolerance,
        )
        step_finite = jnp.isfinite(context.duration) & (context.duration > 0.0)
        step_valid = step_finite & (context.duration <= policy.maximum_step_size)
        source_valid = source_evaluation.valid
        nonlinear_valid = nonlinear.successful
        material_valid = (
            candidate_evaluation.stretch_shear_material_result.evidence.valid
            & candidate_evaluation.bend_twist_material_result.evidence.valid
        )
        candidate_valid = candidate_evaluation.valid & nonlinear.auxiliary.valid
        finite = (
            source_evaluation.finite
            & candidate_evaluation.finite
            & nonlinear.auxiliary.finite
            & jnp.all(jnp.isfinite(q1))
            & jnp.all(jnp.isfinite(v1))
            & ledger.finite
        )
        successful = (
            step_valid
            & source_valid
            & nonlinear_valid
            & material_valid
            & candidate_valid
            & ledger.valid
            & finite
        )
        status = _status(
            step_valid=step_valid,
            source_valid=source_valid,
            linear_valid=jnp.asarray(True),
            nonlinear_attempted=True,
            nonlinear_valid=nonlinear_valid,
            material_valid=material_valid,
            candidate_valid=candidate_valid,
            ledger_valid=ledger.valid & finite,
        )
        backend = jnp.asarray(nonlinear.status, dtype=jnp.int32)
        evidence = ReducedRodStepEvidence(
            source_evaluation,
            candidate_evaluation,
            nonlinear.diagnostics,
            nonlinear,
            ledger,
            step_finite,
            step_valid,
            source_valid,
            nonlinear.diagnostics.final_linear_converged,
            nonlinear_valid,
            material_valid,
            candidate_valid,
            finite,
            successful,
            status,
            backend,
            policy.route,
            policy.policy_id,
        )
        return ReducedRodStepResult(
            integration_source,
            candidate,
            _selected_state(successful, candidate, integration_source),
            jnp.asarray(True),
            successful,
            status,
            backend,
            evidence,
            policy.policy_id,
        )

    def _mechanical_integration(
        self,
        source: TendonDrivenRodPlantState,
        actuator: TendonActuatorStateBank,
        command: TendonDrivenRodPlantCommand,
        context: PlantStepContext,
        /,
    ) -> ReducedRodStepResult:
        if isinstance(self.base_plant.policy, ReducedRodSemiImplicitVelocityEuler):
            return self._semi_implicit_integration(source, actuator, command, context)
        if isinstance(self.base_plant.policy, ReducedRodImplicitMidpoint):
            return self._implicit_midpoint_integration(source, actuator, command, context)
        raise TypeError("The tendon plant has no prepared reduced integrator route.")

    def _actuation_ledger(
        self,
        source: TendonDrivenRodPlantState,
        candidate_reduced: ReducedRodState,
        actuator: TendonActuatorStateBank,
        payout: tuple[TendonActuationEvaluation, ...],
        step: Array,
        /,
    ) -> TendonDrivenRodActuationLedger:
        loaded = self._loaded_evaluations(source.reduced_state, actuator)
        candidate = self._loaded_evaluations(candidate_reduced, actuator)
        source_tension = jnp.stack(tuple(evaluation.tension for evaluation in payout))
        candidate_tension = jnp.stack(
            tuple(evaluation.tension for evaluation in candidate)
        )
        source_energy = jnp.stack(
            tuple(evaluation.stored_energy for evaluation in payout)
        )
        candidate_energy = jnp.stack(
            tuple(evaluation.stored_energy for evaluation in candidate)
        )
        spool_work = jnp.stack(tuple(evaluation.spool_work for evaluation in payout))
        rod_work = (
            0.5
            * step
            * jnp.stack(
                tuple(
                    left.rod_power + right.rod_power
                    for left, right in zip(loaded, candidate, strict=True)
                )
            )
        )
        residual = candidate_energy - source_energy + spool_work + rod_work
        scale = jnp.maximum(
            1.0,
            jnp.max(
                jnp.abs(
                    jnp.concatenate(
                        (source_energy, candidate_energy, spool_work, rod_work)
                    )
                )
            ),
        )
        tolerance = jnp.max(
            jnp.asarray(
                tuple(tendon.plan.power_tolerance for tendon in self.tendons),
                dtype=scale.dtype,
            )
        )
        finite = (
            _all(tuple(evaluation.finite for evaluation in payout))
            & _all(tuple(evaluation.finite for evaluation in loaded))
            & _all(tuple(evaluation.finite for evaluation in candidate))
            & jnp.all(jnp.isfinite(residual))
        )
        source_valid = _all(tuple(value.valid for value in payout)) & _all(
            tuple(value.valid for value in loaded)
        )
        candidate_valid = _all(tuple(value.valid for value in candidate))
        balanced = finite & jnp.all(jnp.abs(residual) <= tolerance * scale)
        valid = finite & source_valid & candidate_valid & balanced
        return TendonDrivenRodActuationLedger(
            payout,
            loaded,
            candidate,
            source_tension,
            candidate_tension,
            source_energy,
            candidate_energy,
            spool_work,
            rod_work,
            residual,
            jnp.sum(spool_work),
            jnp.sum(rod_work),
            jnp.sum(residual),
            finite,
            source_valid,
            candidate_valid,
            balanced,
            valid,
            self.tendon_ids,
        )

    def _mechanics_runtime(
        self,
        payload: TendonDrivenRodPlantState,
        time: Array,
        step_index: Array,
        key: Array,
        /,
    ) -> PlantRuntimeState:
        mechanics = ReducedRodPlantState(
            payload.reduced_state,
            payload.material_state,
            self.base_plant.initial_state.actuator_state,
            self.base_plant.initial_state.contact_state,
            self.base_plant.initial_state.sensor_state,
        )
        return PlantRuntimeState(
            mechanics,
            time,
            step_index,
            key,
            self.base_plant.semantic_provenance.semantic_id,
            self.base_plant.numeric_revision.revision_id,
            self.base_plant.state_schema.schema_id,
            self.base_plant.execution_signature.signature_id,
        )

    def _candidate_observation(
        self,
        payload: TendonDrivenRodPlantState,
        time: Array,
        step_index: Array,
        key: Array,
        /,
    ) -> tuple[
        SoftRobotObservation | None,
        ReducedRodPassiveSensorState | SoftSensorState,
    ]:
        from ..robotics._soft_observations import SoftSensorState

        if self.observation_plan is None:
            return None, payload.sensor_state
        runtime = self._mechanics_runtime(payload, time, step_index, key)
        tendon_state = None
        if (
            self.observation_plan.tendon is not None
            and self.observation_plan.tendon.plan.requires_actuator_state
        ):
            tendon_state = self.observation_plan.bind_tendon_state(
                runtime, payload.actuator_state.states
            )
        sensor_state = (
            payload.sensor_state
            if isinstance(payload.sensor_state, SoftSensorState)
            else None
        )
        observation, candidate_sensor = self.observation_plan.observe(
            runtime,
            tendon_state=tendon_state,
            sensor_state=sensor_state,
        )
        return (
            observation,
            payload.sensor_state if candidate_sensor is None else candidate_sensor,
        )

    def propose_reset(
        self,
        keys: Array,
        parameters: Any,
        /,
        *,
        case_shape: tuple[int, ...],
        initial_time: Array,
    ) -> PlantProposal:
        del parameters
        if case_shape != ():
            raise ValueError("Tendon rod plants support one scalar case.")
        evaluations = self._loaded_evaluations(
            self.initial_state.reduced_state,
            self.initial_state.actuator_state,
        )
        observation, sensor = self._candidate_observation(
            self.initial_state,
            initial_time,
            jnp.asarray(0, dtype=jnp.int32),
            keys,
        )
        candidate = TendonDrivenRodPlantState(
            self.initial_state.reduced_state,
            self.initial_state.material_state,
            self.initial_state.actuator_state,
            self.initial_state.contact_state,
            sensor,
        )
        observation_valid = (
            jnp.asarray(True) if observation is None else observation.valid
        )
        finite = _all(tuple(value.finite for value in evaluations)) & (
            jnp.asarray(True) if observation is None else observation.finite
        )
        source_valid = _all(tuple(value.valid for value in evaluations))
        valid = finite & source_valid & observation_valid
        accepted = self.state_schema.select_cases(valid, candidate, self.initial_state)
        status = jnp.asarray(TendonDrivenRodPlantStatus.SUCCESS, dtype=jnp.int32)
        status = jnp.where(
            ~observation_valid,
            int(TendonDrivenRodPlantStatus.SENSOR_INVALID),
            status,
        )
        status = jnp.where(
            ~source_valid,
            int(TendonDrivenRodPlantStatus.SOURCE_INVALID),
            status,
        ).astype(jnp.int32)
        status = jnp.where(valid, int(TendonDrivenRodPlantStatus.SUCCESS), status).astype(
            jnp.int32
        )
        evidence = TendonDrivenRodPlantResetEvidence(
            evaluations,
            observation,
            observation_valid,
            finite,
            valid,
            status,
            self.plant_id,
        )
        return PlantProposal(
            candidate,
            accepted,
            jnp.asarray(True),
            valid,
            status,
            jnp.asarray(0, dtype=jnp.int32),
            evidence,
        )

    def propose_step(
        self,
        context: PlantStepContext,
        source: Any,
        commands: Any,
        parameters: Any,
        keys: Array,
        /,
    ) -> PlantProposal:
        del parameters
        if not isinstance(source, TendonDrivenRodPlantState):
            raise TypeError("source must be TendonDrivenRodPlantState.")
        if not isinstance(commands, TendonDrivenRodPlantCommand):
            raise TypeError("commands must be TendonDrivenRodPlantCommand.")
        self.state_schema.validate(source)
        self.control_schema.validate(commands)
        within_bounds = self.command_bounds.contains(commands)
        actuator, payout = self._payout(
            source, commands.tendon_commands, context.duration
        )
        integration = self._mechanical_integration(source, actuator, commands, context)
        candidate_reduced = integration.candidate_state.reduced_state
        tendon_ledger = self._actuation_ledger(
            source,
            candidate_reduced,
            actuator,
            payout,
            context.duration,
        )
        pre_observation = TendonDrivenRodPlantState(
            candidate_reduced,
            integration.candidate_state.material_state,
            actuator,
            source.contact_state,
            source.sensor_state,
        )
        observation, sensor = self._candidate_observation(
            pre_observation,
            context.target_time,
            context.step_index + jnp.asarray(1, dtype=jnp.int32),
            keys,
        )
        candidate = TendonDrivenRodPlantState(
            candidate_reduced,
            integration.candidate_state.material_state,
            actuator,
            source.contact_state,
            sensor,
        )
        observation_valid = (
            jnp.asarray(True) if observation is None else observation.valid
        )
        observation_finite = (
            jnp.asarray(True) if observation is None else observation.finite
        )
        finite = integration.evidence.finite & tendon_ledger.finite & observation_finite
        valid = (
            integration.successful
            & within_bounds
            & tendon_ledger.valid
            & observation_valid
            & finite
        )
        accepted = self.state_schema.select_cases(valid, candidate, source)
        status = integration.status
        status = jnp.where(
            ~observation_valid,
            int(TendonDrivenRodPlantStatus.SENSOR_INVALID),
            status,
        )
        status = jnp.where(
            ~tendon_ledger.balanced,
            int(TendonDrivenRodPlantStatus.TENDON_LEDGER_INVALID),
            status,
        )
        status = jnp.where(
            ~tendon_ledger.candidate_valid,
            int(TendonDrivenRodPlantStatus.TENDON_CANDIDATE_INVALID),
            status,
        )
        status = jnp.where(
            ~tendon_ledger.source_valid,
            int(TendonDrivenRodPlantStatus.TENDON_SOURCE_INVALID),
            status,
        )
        status = jnp.where(
            ~within_bounds,
            int(TendonDrivenRodPlantStatus.COMMAND_OUT_OF_BOUNDS),
            status,
        ).astype(jnp.int32)
        status = jnp.where(valid, int(TendonDrivenRodPlantStatus.SUCCESS), status).astype(
            jnp.int32
        )
        evidence = TendonDrivenRodPlantEvidence(
            integration,
            tendon_ledger,
            observation,
            within_bounds,
            observation_valid,
            finite,
            valid,
            status,
            self.plant_id,
            self.tendon_ids,
        )
        return PlantProposal(
            candidate,
            accepted,
            jnp.asarray(True),
            valid,
            status,
            integration.backend_status,
            evidence,
        )

    def mechanics_runtime_view(self, state: PlantRuntimeState, /) -> PlantRuntimeState:
        if not isinstance(state.payload, TendonDrivenRodPlantState):
            raise TypeError("Runtime payload must be TendonDrivenRodPlantState.")
        payload = ReducedRodPlantState(
            state.payload.reduced_state,
            state.payload.material_state,
            self.base_plant.initial_state.actuator_state,
            self.base_plant.initial_state.contact_state,
            self.base_plant.initial_state.sensor_state,
        )
        return PlantRuntimeState(
            payload,
            state.time,
            state.step_index,
            state.key,
            self.base_plant.semantic_provenance.semantic_id,
            self.base_plant.numeric_revision.revision_id,
            self.base_plant.state_schema.schema_id,
            self.base_plant.execution_signature.signature_id,
        )

    def mass_response(
        self, state: PlantRuntimeState, /
    ) -> TendonDrivenRodMassResponseRevision:
        base = self.base_plant.mass_response(self.mechanics_runtime_view(state))
        return TendonDrivenRodMassResponseRevision(
            base,
            state.time,
            state.step_index,
            self.plant_id,
            self.semantic_provenance.semantic_id,
            self.numeric_revision.revision_id,
            self.state_schema.schema_id,
            self.execution_signature.signature_id,
        )


TendonDrivenRodPlant = PreparedTendonDrivenRodPlant


def prepare_tendon_driven_rod_plant(
    base_plant: PreparedReducedRodPlant,
    tendons: Sequence[PreparedFrictionlessElasticTendon],
    initial_free_lengths: Sequence[ArrayLike],
    /,
    **kwargs: Any,
) -> PreparedTendonDrivenRodPlant:
    return PreparedTendonDrivenRodPlant(
        base_plant, tendons, initial_free_lengths, **kwargs
    )


__all__ = [
    "prepare_tendon_driven_rod_plant",
    "PreparedTendonDrivenRodPlant",
    "TendonActuatorStateBank",
    "TendonDrivenRodActuationLedger",
    "TendonDrivenRodCommandBounds",
    "TendonDrivenRodMassResponseRevision",
    "TendonDrivenRodPlant",
    "TendonDrivenRodPlantCommand",
    "TendonDrivenRodPlantEvidence",
    "TendonDrivenRodPlantParameters",
    "TendonDrivenRodPlantResetEvidence",
    "TendonDrivenRodPlantState",
    "TendonDrivenRodPlantStatus",
]
