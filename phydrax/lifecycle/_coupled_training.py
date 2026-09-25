#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""In-situ training coupled to a discrete plant as one joint transaction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any, final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from .._training_kernel import (
    CommittedUpdateHook,
    enforce_rejection_budget,
    PreparedTrainingKernel,
    TrainingAttemptEvidence,
    TrainingAttemptOutcome,
    TrainingKernelState,
)
from .._tree_math import tree_where
from ._transaction import commit_candidate, TransactionalCandidate


if TYPE_CHECKING:
    from ..dynamics._plant import (
        AbstractDiscretePlant,
        PlantParameters,
        PlantRuntimeState,
        PlantStepContext,
        PlantStepResult,
    )


class CoupledTrainingPolicy(StrEnum):
    """What a finite training rejection means for a valid physical step.

    `PHYSICAL_MAY_COMMIT` commits the plant's own accepted step with unchanged
    parameters; `JOINTLY_REQUIRED` rejects the physical step as well.
    """

    PHYSICAL_MAY_COMMIT = "physical-may-commit"
    JOINTLY_REQUIRED = "jointly-required"


@final
class CoupledTrainingEvidence(StrictModule):
    """Joint decision of one coupled step.

    `physical_accepted` is the global acceptance over contributing (attempted)
    cases for shared parameters, or the per-lane acceptance for per-lane
    parameters. `physical_committed` marks committed plant cases and
    `training_committed` whether the kernel state of this attempt committed.
    """

    physical_successful: Array
    physical_accepted: Array
    physical_committed: Array
    training_committed: Array
    training: TrainingAttemptEvidence
    plant_status: Array
    plant_evidence: Any


@final
class CoupledTrainingResult(StrictModule):
    plant_state: PlantRuntimeState
    kernel_state: TrainingKernelState
    evidence: CoupledTrainingEvidence


PayloadFunction = Callable[["PlantRuntimeState", "PlantStepResult"], PyTree[Any]]


def _select_lanes(
    selector: Array, candidate: PyTree[Any], source: PyTree[Any], /
) -> PyTree[Any]:
    """Select whole lanes (leading axis) of congruent lane-stacked trees."""
    return jax.tree.map(
        lambda new, old: jnp.where(
            jnp.reshape(selector, selector.shape + (1,) * (new.ndim - selector.ndim)),
            new,
            old,
        ),
        candidate,
        source,
    )


def _physical_acceptance(
    kernel: PreparedTrainingKernel,
    kernel_state: TrainingKernelState,
    step: PlantStepResult,
    /,
) -> Array:
    case_shape = step.successful.shape
    if not kernel.lane_parameters:
        # Shared parameters learn from every contributing case, so every
        # attempted case must be physically accepted.
        return jnp.all(step.successful | ~step.attempted) & jnp.any(step.attempted)
    lanes = kernel_state.attempt_cursor.shape
    if case_shape != lanes or kernel.lane_layout.kind == "item":
        raise ValueError(
            f"{kernel.context}: per-lane parameters need a case or member lane layout "
            f"aligned with the plant cases; plant case shape {case_shape}, "
            f"{kernel.lane_layout.kind} lanes {lanes}."
        )
    return step.successful


def _coupled_transaction(
    plant: AbstractDiscretePlant,
    plant_state: PlantRuntimeState,
    kernel: PreparedTrainingKernel,
    kernel_state: TrainingKernelState,
    payload_fn: PayloadFunction,
    policy: CoupledTrainingPolicy,
    context: PlantStepContext,
    commands: PyTree[Any] | None,
    plant_parameters: PlantParameters,
    /,
) -> tuple[PlantRuntimeState, TrainingKernelState, CoupledTrainingEvidence]:
    # Imported lazily: importing the dynamics package reaches qualification,
    # which imports this lifecycle package.
    from ..dynamics._plant import _select_runtime_state

    step = plant.step(context, plant_state, commands, plant_parameters)
    payload = payload_fn(plant_state, step)
    physical_accepted = _physical_acceptance(kernel, kernel_state, step)
    trained, training = kernel.attempt(kernel_state, payload)
    training_accepted = training.outcome == TrainingAttemptOutcome.ACCEPTED
    case_shape = step.successful.shape
    match policy:
        case CoupledTrainingPolicy.PHYSICAL_MAY_COMMIT:
            physical_committed = jnp.ones(case_shape, dtype=jnp.bool_)
        case CoupledTrainingPolicy.JOINTLY_REQUIRED:
            physical_committed = jnp.broadcast_to(
                physical_accepted & training_accepted, case_shape
            )
        case _:
            raise ValueError(f"Unknown coupled training policy {policy!r}.")
    # A training candidate derived from a rejected physical step never commits;
    # otherwise the kernel's own outcome state (accepted, rule-authorized
    # rejection, or rollback) commits.
    training_committed = physical_accepted
    plant_proposed = _select_runtime_state(
        plant.state_schema,
        physical_committed,
        step.accepted_state,
        plant_state,
        case_shape,
    )
    kernel_proposed = (
        _select_lanes(training_committed, trained, kernel_state)
        if kernel.lane_parameters
        else tree_where(training_committed, trained, kernel_state)
    )
    evidence = CoupledTrainingEvidence(
        physical_successful=step.successful,
        physical_accepted=physical_accepted,
        physical_committed=physical_committed & step.successful,
        training_committed=training_committed,
        training=training,
        plant_status=step.status,
        plant_evidence=step.evidence,
    )
    joint = commit_candidate(
        TransactionalCandidate(
            (plant_state, kernel_state),
            (plant_proposed, kernel_proposed),
            evidence,
            jnp.any(physical_committed) | jnp.any(training_committed),
            f"coupled-training:{kernel.checkpoint_id}",
        )
    )
    committed_plant, committed_kernel = joint.state
    return committed_plant, committed_kernel, joint.evidence


_compiled_transaction = eqx.filter_jit(_coupled_transaction)


def coupled_training_step(
    plant: AbstractDiscretePlant,
    plant_state: PlantRuntimeState,
    kernel: PreparedTrainingKernel,
    kernel_state: TrainingKernelState,
    payload_fn: PayloadFunction,
    policy: CoupledTrainingPolicy | str,
    /,
    *,
    context: PlantStepContext,
    commands: PyTree[Any] | None,
    plant_parameters: PlantParameters,
    hooks: Sequence[CommittedUpdateHook] = (),
) -> CoupledTrainingResult:
    """Advance a plant and train on its step as one atomic joint transaction.

    Decision order: the plant proposes its physical candidate;
    `payload_fn(source, step)` derives the training payload only from the
    committed source and that candidate; the physical acceptance is evaluated;
    the kernel attempt yields its training outcome; `policy` decides; one
    `TransactionalCandidate` commits plant and kernel state together. Physical
    rejection always discards the derived training update (parameters, rule
    state, model state, targets, and cursors are restored exactly). Shared
    parameters require acceptance of every attempted case; per-lane parameters
    need a lane layout aligned with the plant cases and commit lane by lane.
    Hooks and the rejection-budget check run only after the joint commit.
    """
    from ..dynamics._plant import (
        AbstractDiscretePlant,
        PlantParameters,
        PlantRuntimeState,
        PlantStepContext,
    )

    policy_ = CoupledTrainingPolicy(policy)
    if not isinstance(plant, AbstractDiscretePlant):
        raise TypeError("plant must be an AbstractDiscretePlant.")
    if not isinstance(plant_state, PlantRuntimeState):
        raise TypeError("plant_state must be a PlantRuntimeState.")
    if not isinstance(kernel, PreparedTrainingKernel):
        raise TypeError("kernel must be a PreparedTrainingKernel.")
    if not isinstance(kernel_state, TrainingKernelState):
        raise TypeError("kernel_state must be a TrainingKernelState.")
    if not callable(payload_fn):
        raise TypeError("payload_fn must be callable.")
    if not isinstance(context, PlantStepContext):
        raise TypeError("context must be a PlantStepContext.")
    if not isinstance(plant_parameters, PlantParameters):
        raise TypeError("plant_parameters must be PlantParameters.")
    plant_next, kernel_next, evidence = _compiled_transaction(
        plant,
        plant_state,
        kernel,
        kernel_state,
        payload_fn,
        policy_,
        context,
        commands,
        plant_parameters,
    )
    committed, outcome, consecutive, attempt = jax.device_get(
        (
            evidence.training_committed,
            evidence.training.outcome,
            kernel_next.consecutive_rejections,
            kernel_next.attempt_cursor,
        )
    )
    enforce_rejection_budget(kernel, outcome, consecutive, attempt)
    if np.any(committed & (outcome == TrainingAttemptOutcome.ACCEPTED)):
        for hook in hooks:
            hook(kernel_next, evidence.training)
    return CoupledTrainingResult(plant_next, kernel_next, evidence)


__all__ = [
    "coupled_training_step",
    "CoupledTrainingEvidence",
    "CoupledTrainingPolicy",
    "CoupledTrainingResult",
]
