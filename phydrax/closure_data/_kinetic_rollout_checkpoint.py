#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from .._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from .._model import AbstractArrayModel
from ._kinetic_rollout import PreparedSmoothCompressibleRolloutDataset
from ._kinetic_rollout_training import (
    _model_structure_id,
    _validate_dataset,
    _validate_state_binding,
    KineticRolloutTrainingPlan,
    KineticRolloutTrainingState,
)


_KIND = "kinetic-rollout-training-checkpoint"
_FIELDS = {
    "kind",
    "plan_id",
    "dataset_id",
    "runtime_id",
    "stage_one_artifact_id",
    "support_id",
    "normalizer_id",
    "model_structure_id",
    "state_id",
    "progress",
    "model",
    "optimizer_state",
    "best_model",
    "key",
    "best_loss",
    "arrays",
}
_PROGRESS_FIELDS = {
    "attempt_count",
    "accepted_update_count",
    "rejection_count",
    "curriculum_index",
    "accepted_in_curriculum",
    "training_cursor",
    "guard_cursor",
    "last_update_accepted",
}


def _progress(state: KineticRolloutTrainingState, /) -> dict[str, int | bool]:
    return {
        "attempt_count": state.attempt_count,
        "accepted_update_count": state.accepted_update_count,
        "rejection_count": state.rejection_count,
        "curriculum_index": state.curriculum_index,
        "accepted_in_curriculum": state.accepted_in_curriculum,
        "training_cursor": state.training_cursor,
        "guard_cursor": state.guard_cursor,
        "last_update_accepted": state.last_update_accepted,
    }


def write_kinetic_rollout_checkpoint(
    path: str | Path,
    state: KineticRolloutTrainingState,
    plan: KineticRolloutTrainingPlan,
    dataset: PreparedSmoothCompressibleRolloutDataset,
    /,
) -> Path:
    """Atomically write an exact pickle-free transactional training boundary."""

    if not isinstance(state, KineticRolloutTrainingState):
        raise TypeError("state must be KineticRolloutTrainingState.")
    if not isinstance(plan, KineticRolloutTrainingPlan):
        raise TypeError("plan must be KineticRolloutTrainingPlan.")
    _validate_state_binding(state, plan, dataset)
    arrays: dict[str, object] = {}
    manifest: dict[str, Any] = {
        "kind": _KIND,
        "plan_id": state.plan_id,
        "dataset_id": state.dataset_id,
        "runtime_id": plan.dynamics.prepared_id,
        "stage_one_artifact_id": plan.binding_plan.parent_artifact_id,
        "support_id": plan.binding_plan.support.support_id,
        "normalizer_id": plan.binding_plan.normalizer.normalizer_id,
        "model_structure_id": state.model_structure_id,
        "state_id": state.state_id,
        "progress": _progress(state),
        "model": pack_array_tree(
            "model", eqx.filter(state.model, eqx.is_inexact_array), arrays
        ),
        "optimizer_state": pack_array_tree(
            "optimizer_state", state.optimizer_state, arrays
        ),
        "best_model": pack_array_tree(
            "best_model", eqx.filter(state.best_model, eqx.is_inexact_array), arrays
        ),
        "key": pack_array_tree("key", state.key, arrays),
        "best_loss": pack_array_tree("best_loss", state.best_loss, arrays),
    }
    return write_array_archive(path, manifest=manifest, arrays=arrays)


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
    manifest, arrays = read_array_archive(path)
    if set(manifest) != _FIELDS or manifest.get("kind") != _KIND:
        raise ValueError("Kinetic-rollout checkpoint manifest is not canonical.")
    if (
        manifest.get("plan_id") != plan.plan_id
        or manifest.get("dataset_id") != dataset.preparation_id
        or manifest.get("runtime_id") != plan.dynamics.prepared_id
        or manifest.get("stage_one_artifact_id") != plan.binding_plan.parent_artifact_id
        or manifest.get("support_id") != plan.binding_plan.support.support_id
        or manifest.get("normalizer_id") != plan.binding_plan.normalizer.normalizer_id
        or manifest.get("model_structure_id") != structure_id
    ):
        raise ValueError(
            "Kinetic-rollout checkpoint plan, dataset, or model identity mismatch."
        )
    progress = manifest.get("progress")
    if not isinstance(progress, dict) or set(progress) != _PROGRESS_FIELDS:
        raise ValueError("Kinetic-rollout checkpoint progress is incomplete.")
    last_accepted = progress["last_update_accepted"]
    if not isinstance(last_accepted, bool):
        raise ValueError("Checkpoint acceptance marker must be boolean.")

    model_arrays_template, model_static = eqx.partition(
        model_template, eqx.is_inexact_array
    )
    optimizer_template = plan.optimizer().init(model_arrays_template)
    model_arrays = unpack_array_tree(manifest["model"], arrays, model_arrays_template)
    optimizer_state = unpack_array_tree(
        manifest["optimizer_state"], arrays, optimizer_template
    )
    best_model_arrays = unpack_array_tree(
        manifest["best_model"], arrays, model_arrays_template
    )
    model = eqx.combine(model_arrays, model_static)
    best_model = eqx.combine(best_model_arrays, model_static)
    restored_key = unpack_array_tree(
        manifest["key"], arrays, jnp.zeros((2,), dtype=jnp.uint32)
    )
    best_loss = unpack_array_tree(
        manifest["best_loss"],
        arrays,
        jnp.zeros((), dtype=jnp.dtype(dataset.trajectories[0].schema.dtype)),
    )
    state = KineticRolloutTrainingState(
        model,
        optimizer_state,
        best_model,
        restored_key,
        best_loss,
        attempt_count=_integer(progress, "attempt_count"),
        accepted_update_count=_integer(progress, "accepted_update_count"),
        rejection_count=_integer(progress, "rejection_count"),
        curriculum_index=_integer(progress, "curriculum_index"),
        accepted_in_curriculum=_integer(progress, "accepted_in_curriculum"),
        training_cursor=_integer(progress, "training_cursor"),
        guard_cursor=_integer(progress, "guard_cursor"),
        last_update_accepted=last_accepted,
        plan_id=plan.plan_id,
        dataset_id=dataset.preparation_id,
        model_structure_id=structure_id,
    )
    if state.state_id != manifest.get("state_id"):
        raise ValueError("Kinetic-rollout checkpoint content identity is corrupt.")
    _validate_state_binding(state, plan, dataset)
    return state


__all__ = [
    "read_kinetic_rollout_checkpoint",
    "write_kinetic_rollout_checkpoint",
]
