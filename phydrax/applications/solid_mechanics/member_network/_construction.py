#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._buckling import tangent_stability, TangentStabilityResult
from ._equilibrium import (
    MemberNetworkInputs,
    MemberNetworkPlan,
    MemberNetworkProblem,
    MemberNetworkResult,
    plan_member_network,
    prepare_member_network,
    solve_member_network,
)
from ._reference import MemberKinematics


class InstallationRule(IntEnum):
    DECLARED_STRESS_FREE_LENGTH = 0
    STRESS_FREE_AT_CURRENT_GEOMETRY = 1
    DECLARED_INITIAL_STRAIN = 2
    ACTUATOR_CONTROLLED = 3


class LoadOperation(IntEnum):
    REPLACE = 0
    ADD = 1
    REMOVE = 2
    RAMP = 3


class ActuatorOperation(IntEnum):
    NONE = 0
    APPLY_FORCE_HOLD = 1
    APPLY_STROKE_HOLD = 2
    CHANGE_STRESS_FREE_LENGTH = 3
    RELEASE_FORCE_CONTROL = 4
    LOCK_OFF = 5


class ConstructionStage(StrictModule, NonTrainableState):
    """One fully declared topology, support, load, and installation stage."""

    problem: MemberNetworkProblem
    inputs: MemberNetworkInputs
    installation_rule: Array
    actuator_operation: Array
    load_operation: LoadOperation = eqx.field(static=True)
    load_factor: float = eqx.field(static=True)
    require_tangent_stability: bool = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: MemberNetworkProblem,
        inputs: MemberNetworkInputs,
        installation_rule: ArrayLike,
        /,
        *,
        actuator_operation: ArrayLike | None = None,
        load_operation: LoadOperation = LoadOperation.REPLACE,
        load_factor: float = 1.0,
        require_tangent_stability: bool = False,
        stage_id: str,
    ):
        rules = jnp.asarray(installation_rule, dtype=jnp.int32)
        count = problem.definition.structure.member_count
        if rules.shape != (count,) or bool(
            jnp.any(rules < int(InstallationRule.DECLARED_STRESS_FREE_LENGTH))
            | jnp.any(rules > int(InstallationRule.ACTUATOR_CONTROLLED))
        ):
            raise ValueError("installation_rule must contain one valid rule per member.")
        actuator = (
            jnp.zeros((count,), dtype=jnp.int32)
            if actuator_operation is None
            else jnp.asarray(actuator_operation, dtype=jnp.int32)
        )
        if actuator.shape != (count,) or bool(
            jnp.any(actuator < int(ActuatorOperation.NONE))
            | jnp.any(actuator > int(ActuatorOperation.LOCK_OFF))
        ):
            raise ValueError(
                "actuator_operation must contain one valid value per member."
            )
        identifier = str(stage_id)
        if not identifier:
            raise ValueError("stage_id must be nonempty.")
        self.problem = problem
        self.inputs = inputs
        self.installation_rule = rules
        self.actuator_operation = actuator
        self.load_operation = LoadOperation(load_operation)
        self.load_factor = float(load_factor)
        self.require_tangent_stability = bool(require_tangent_stability)
        self.stage_id = identifier


class ConstructionSequencePlan(StrictModule, NonTrainableState):
    stages: tuple[ConstructionStage, ...]
    member_plans: tuple[MemberNetworkPlan, ...]
    sequence_id: str = eqx.field(static=True)


class ConstructionStageResult(StrictModule):
    stage_id: str = eqx.field(static=True)
    equilibrium: MemberNetworkResult
    stability: TangentStabilityResult | None
    installed_rest_lengths: Array
    strain_energy: Array
    external_work: Array
    successful: Array


class ConstructionCheckpoint(StrictModule):
    completed_stage: int = eqx.field(static=True)
    kinematics: MemberKinematics
    rest_lengths: Array
    sequence_id: str = eqx.field(static=True)


class ConstructionSequenceResult(StrictModule):
    stages: tuple[ConstructionStageResult, ...]
    first_failed_stage: Array
    final_kinematics: MemberKinematics
    checkpoint: ConstructionCheckpoint
    successful: Array
    sequence_id: str = eqx.field(static=True)


def _transfer_kinematics(
    previous_problem: MemberNetworkProblem,
    current_problem: MemberNetworkProblem,
    previous: MemberKinematics,
    /,
) -> MemberKinematics:
    old_structure = previous_problem.definition.structure
    new_structure = current_problem.definition.structure
    old_node_index = {
        identifier: index for index, identifier in enumerate(old_structure.node_ids)
    }
    positions = current_problem.definition.reference.positions
    rotations = current_problem.definition.reference.rotation_vectors
    for new_index, identifier in enumerate(new_structure.node_ids):
        if identifier in old_node_index:
            old_index = old_node_index[identifier]
            positions = positions.at[new_index].set(previous.positions[old_index])
            rotations = rotations.at[new_index].set(previous.rotation_vectors[old_index])
    return MemberKinematics(positions, rotations)


def _stage_inputs(
    stage: ConstructionStage,
    initial: MemberKinematics,
    previous_inputs: MemberNetworkInputs | None,
    /,
) -> MemberNetworkInputs:
    inputs = stage.inputs
    structure = stage.problem.definition.structure
    vectors = (
        initial.positions[structure.receivers] - initial.positions[structure.senders]
    )
    current_lengths = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
    rest_lengths = jnp.where(
        stage.installation_rule == int(InstallationRule.STRESS_FREE_AT_CURRENT_GEOMETRY),
        current_lengths,
        inputs.rest_lengths,
    )
    forces = inputs.nodal_forces
    moments = inputs.nodal_moments
    if previous_inputs is not None and previous_inputs.nodal_forces.shape == forces.shape:
        if stage.load_operation == LoadOperation.ADD:
            forces = previous_inputs.nodal_forces + stage.load_factor * forces
            moments = previous_inputs.nodal_moments + stage.load_factor * moments
        elif stage.load_operation == LoadOperation.REMOVE:
            forces = previous_inputs.nodal_forces - stage.load_factor * forces
            moments = previous_inputs.nodal_moments - stage.load_factor * moments
        elif stage.load_operation == LoadOperation.RAMP:
            forces = (
                1.0 - stage.load_factor
            ) * previous_inputs.nodal_forces + stage.load_factor * forces
            moments = (
                1.0 - stage.load_factor
            ) * previous_inputs.nodal_moments + stage.load_factor * moments
    return eqx.tree_at(
        lambda selected: (
            selected.rest_lengths,
            selected.nodal_forces,
            selected.nodal_moments,
        ),
        inputs,
        (rest_lengths, forces, moments),
    )


def plan_construction_sequence(
    stages: Sequence[ConstructionStage],
    initial_kinematics: MemberKinematics,
    /,
) -> ConstructionSequencePlan:
    stages_ = tuple(stages)
    if not stages_ or any(not isinstance(stage, ConstructionStage) for stage in stages_):
        raise TypeError("stages must contain ConstructionStage values.")
    if len({stage.stage_id for stage in stages_}) != len(stages_):
        raise ValueError("Construction stage IDs must be unique.")
    plans = []
    current = initial_kinematics
    previous_problem = stages_[0].problem
    previous_inputs = None
    for stage in stages_:
        if stage is not stages_[0]:
            current = _transfer_kinematics(previous_problem, stage.problem, current)
        inputs = _stage_inputs(stage, current, previous_inputs)
        plans.append(plan_member_network(stage.problem, inputs, current))
        previous_problem = stage.problem
        previous_inputs = inputs
    sequence_id = canonical_fingerprint(
        {
            "kind": "construction-sequence",
            "stages": [stage.stage_id for stage in stages_],
            "plans": [plan.plan_id for plan in plans],
        }
    )
    return ConstructionSequencePlan(stages_, tuple(plans), sequence_id)


def solve_construction_sequence(
    plan: ConstructionSequencePlan,
    initial_kinematics: MemberKinematics,
    /,
    *,
    start_checkpoint: ConstructionCheckpoint | None = None,
) -> ConstructionSequenceResult:
    if start_checkpoint is not None:
        if start_checkpoint.sequence_id != plan.sequence_id:
            raise ValueError("Construction checkpoint belongs to another sequence.")
        start = start_checkpoint.completed_stage + 1
        current = start_checkpoint.kinematics
        previous_inputs = plan.stages[start_checkpoint.completed_stage].inputs
    else:
        start = 0
        current = initial_kinematics
        previous_inputs = None
    results: list[ConstructionStageResult] = []
    first_failed = jnp.asarray(-1, dtype=jnp.int32)
    previous_problem = plan.stages[max(start - 1, 0)].problem
    rest_lengths = (
        start_checkpoint.rest_lengths
        if start_checkpoint is not None
        else plan.stages[0].inputs.rest_lengths
    )
    for index in range(start, len(plan.stages)):
        stage = plan.stages[index]
        if index > start or start > 0:
            current = _transfer_kinematics(previous_problem, stage.problem, current)
        inputs = _stage_inputs(stage, current, previous_inputs)
        prepared = prepare_member_network(plan.member_plans[index], inputs, current)
        equilibrium = solve_member_network(prepared)
        current = equilibrium.state.kinematics
        rest_lengths = inputs.rest_lengths
        stability = (
            tangent_stability(stage.problem, inputs, current)
            if stage.require_tangent_stability and bool(equilibrium.successful)
            else None
        )
        stable = jnp.asarray(True) if stability is None else stability.stable
        successful = equilibrium.successful & stable
        first_failed = jnp.where(
            (first_failed < 0) & ~successful,
            index,
            first_failed,
        )
        external_work = jnp.sum(inputs.nodal_forces * current.positions) + jnp.sum(
            inputs.nodal_moments * current.rotation_vectors
        )
        results.append(
            ConstructionStageResult(
                stage.stage_id,
                equilibrium,
                stability,
                rest_lengths,
                equilibrium.state.assembly.energy,
                external_work,
                successful,
            )
        )
        previous_problem = stage.problem
        previous_inputs = inputs
    completed = len(plan.stages) - 1
    checkpoint = ConstructionCheckpoint(
        completed,
        current,
        rest_lengths,
        plan.sequence_id,
    )
    successful = jnp.asarray(first_failed < 0)
    return ConstructionSequenceResult(
        tuple(results),
        first_failed,
        current,
        checkpoint,
        successful,
        plan.sequence_id,
    )


def enumerate_construction_sequences(
    candidates: Sequence[ConstructionSequencePlan],
    evaluator: Callable[[ConstructionSequencePlan], tuple[Array, Array]],
    /,
) -> tuple[int, Array, tuple[Array, ...]]:
    """Exact bounded enumeration for explicitly supplied sequence catalogs."""
    candidates_ = tuple(candidates)
    if not candidates_ or not callable(evaluator):
        raise TypeError("A nonempty candidate sequence and evaluator are required.")
    scores = []
    valid = []
    for candidate in candidates_:
        score, accepted = evaluator(candidate)
        scores.append(jnp.asarray(score))
        valid.append(jnp.asarray(accepted, dtype=bool))
    score_array = jnp.stack(tuple(scores))
    valid_array = jnp.stack(tuple(valid))
    safe = jnp.where(valid_array, score_array, jnp.inf)
    index = int(jnp.argmin(safe))
    return index, score_array[index], tuple(valid_array)


__all__ = [
    "ActuatorOperation",
    "ConstructionCheckpoint",
    "ConstructionSequencePlan",
    "ConstructionSequenceResult",
    "ConstructionStage",
    "ConstructionStageResult",
    "InstallationRule",
    "LoadOperation",
    "enumerate_construction_sequences",
    "plan_construction_sequence",
    "solve_construction_sequence",
]
