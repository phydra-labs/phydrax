#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr

from .._model import AbstractArrayModel
from .._training_checkpoint import (
    load_training_checkpoint,
    read_training_checkpoint_metadata,
    save_training_checkpoint,
)
from .._training_kernel import build_training_checkpoint
from ._kinetic_rollout import PreparedSmoothCompressibleRolloutDataset
from ._kinetic_rollout_training import (
    _model_structure_id,
    _training_kernel,
    _validate_dataset,
    _validate_state_binding,
    KineticRolloutTrainingPlan,
    KineticRolloutTrainingState,
)


_FORMAT = "kinetic-rollout-training-checkpoint"
_FIELDS = {
    "plan_id",
    "dataset_id",
    "runtime_id",
    "stage_one_artifact_id",
    "support_id",
    "normalizer_id",
    "model_structure_id",
    "state_id",
    "progress",
}
_PROGRESS_FIELDS = {
    "curriculum_index",
    "accepted_in_curriculum",
    "training_cursor",
    "guard_cursor",
    "last_update_accepted",
}


def _identities(
    plan: KineticRolloutTrainingPlan,
    dataset: PreparedSmoothCompressibleRolloutDataset,
    model_structure_id: str,
    /,
) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "dataset_id": dataset.preparation_id,
        "runtime_id": plan.dynamics.prepared_id,
        "stage_one_artifact_id": plan.binding_plan.parent_artifact_id,
        "support_id": plan.binding_plan.support.support_id,
        "normalizer_id": plan.binding_plan.normalizer.normalizer_id,
        "model_structure_id": model_structure_id,
    }


def write_kinetic_rollout_checkpoint(
    path: str | Path,
    state: KineticRolloutTrainingState,
    plan: KineticRolloutTrainingPlan,
    dataset: PreparedSmoothCompressibleRolloutDataset,
    /,
) -> Path:
    """Atomically write an exact pickle-free transactional training boundary.

    `path` is a checkpoint directory holding the training-kernel payload (model
    parameters, Optax and guard state, root key, cursors) plus the frontend's
    best-model selection and curriculum cursors.
    """

    if not isinstance(state, KineticRolloutTrainingState):
        raise TypeError("state must be KineticRolloutTrainingState.")
    if not isinstance(plan, KineticRolloutTrainingPlan):
        raise TypeError("plan must be KineticRolloutTrainingPlan.")
    _validate_state_binding(state, plan, dataset)
    kernel = _training_kernel(state.model, plan, dataset)
    destination = Path(path)
    save_training_checkpoint(
        destination,
        # Every attempt is atomic, so a post-rejection state is a boundary.
        build_training_checkpoint(kernel, state.training, allow_intermediate=True),
        (state.best_model, state.best_loss),
        format=_FORMAT,
        metadata={
            **_identities(plan, dataset, state.model_structure_id),
            "state_id": state.state_id,
            "progress": {
                "curriculum_index": state.curriculum_index,
                "accepted_in_curriculum": state.accepted_in_curriculum,
                "training_cursor": state.training_cursor,
                "guard_cursor": state.guard_cursor,
                "last_update_accepted": state.last_update_accepted,
            },
        },
    )
    return destination


def _integer(progress: dict[str, Any], name: str, /) -> int:
    value = progress[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Checkpoint progress field {name!r} is invalid.")
    return value


def read_kinetic_rollout_checkpoint(
    path: str | Path,
    plan: KineticRolloutTrainingPlan,
    dataset: PreparedSmoothCompressibleRolloutDataset,
    model_template: AbstractArrayModel,
    /,
) -> KineticRolloutTrainingState:
    """Restore arrays only against exact runtime, data, and model templates."""

    if not isinstance(plan, KineticRolloutTrainingPlan):
        raise TypeError("plan must be KineticRolloutTrainingPlan.")
    if not isinstance(model_template, AbstractArrayModel):
        raise TypeError("model_template must be an AbstractArrayModel.")
    _validate_dataset(plan, dataset)
    structure_id = _model_structure_id(model_template)
    metadata = read_training_checkpoint_metadata(path, format=_FORMAT)
    if set(metadata) != _FIELDS:
        raise ValueError("Kinetic-rollout checkpoint manifest is not canonical.")
    if any(
        metadata[name] != value
        for name, value in _identities(plan, dataset, structure_id).items()
    ):
        raise ValueError(
            "Kinetic-rollout checkpoint plan, dataset, or model identity mismatch."
        )
    progress = metadata["progress"]
    if not isinstance(progress, dict) or set(progress) != _PROGRESS_FIELDS:
        raise ValueError("Kinetic-rollout checkpoint progress is incomplete.")
    last_accepted = progress["last_update_accepted"]
    if not isinstance(last_accepted, bool):
        raise ValueError("Checkpoint acceptance marker must be boolean.")

    kernel = _training_kernel(model_template, plan, dataset)
    loaded = load_training_checkpoint(
        path,
        kernel,
        kernel.init(model_template, jr.key(0)),
        (
            model_template,
            jnp.zeros((), dtype=jnp.dtype(dataset.trajectories[0].schema.dtype)),
        ),
        format=_FORMAT,
    )
    training = loaded.restored.state
    best_model, best_loss = loaded.extra
    state = KineticRolloutTrainingState(
        kernel.tree(training),
        training,
        best_model,
        best_loss,
        curriculum_index=_integer(progress, "curriculum_index"),
        accepted_in_curriculum=_integer(progress, "accepted_in_curriculum"),
        training_cursor=_integer(progress, "training_cursor"),
        guard_cursor=_integer(progress, "guard_cursor"),
        last_update_accepted=last_accepted,
        plan_id=plan.plan_id,
        dataset_id=dataset.preparation_id,
        model_structure_id=structure_id,
    )
    if state.state_id != metadata["state_id"]:
        raise ValueError("Kinetic-rollout checkpoint content identity is corrupt.")
    _validate_state_binding(state, plan, dataset)
    return state


__all__ = [
    "read_kinetic_rollout_checkpoint",
    "write_kinetic_rollout_checkpoint",
]
