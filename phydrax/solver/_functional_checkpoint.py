#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from .._model._structure import deserialise_model_leaf, serialise_model_leaf
from .._training import (
    DelayedTargetPolicy,
    ExponentialMovingAverageTargetPolicy,
    TrainingProgress,
)
from .._training_checkpoint import (
    _deserialize_root_key,
    _prune_state_files,
    _publish_manifest,
    _publish_state,
    _read_manifest,
    _serialize_root_key,
    _verify_state,
)
from ._functional_training import FunctionalTrainingPlan, FunctionalTrainingState


_FUNCTIONAL_CHECKPOINT_FORMAT = "phydrax-functional-training-checkpoint"


def _target_policy_contract(state: FunctionalTrainingState, /) -> dict[str, Any] | None:
    if state.target_state is None:
        return None
    policy = state.target_state.policy
    if isinstance(policy, DelayedTargetPolicy):
        return {
            "kind": "delayed",
            "delay": policy.delay,
        }
    if isinstance(policy, ExponentialMovingAverageTargetPolicy):
        return {
            "kind": "exponential-moving-average",
            "decay": policy.decay,
            "start_step": policy.start_step,
            "update_every": policy.update_every,
            "source": policy.source,
        }
    raise TypeError("Functional checkpoint target policy has an unsupported type.")


@dataclass(frozen=True, slots=True)
class FunctionalTrainingCheckpoint:
    functions: Any
    objective: Any
    state: FunctionalTrainingState
    path: Path


def save_functional_training_checkpoint(
    path: str | Path,
    solver: Any,
    state: FunctionalTrainingState,
    plan: FunctionalTrainingPlan,
    /,
) -> Path:
    """Atomically publish one exact accepted-update functional training state."""
    if not isinstance(state, FunctionalTrainingState):
        raise TypeError("state must be a FunctionalTrainingState.")
    if not isinstance(plan, FunctionalTrainingPlan):
        raise TypeError("plan must be a FunctionalTrainingPlan.")
    if state.progress.update_step < 0:
        raise ValueError("Checkpoint progress must be non-negative.")
    if state.target_state is not None and int(state.target_state.update_count) != int(
        state.progress.update_step
    ):
        raise ValueError(
            "Checkpoint target state is not at the accepted-update boundary."
        )
    destination = Path(path)
    state_path, checksum = _publish_state(
        destination,
        lambda target: eqx.tree_serialise_leaves(
            target,
            (solver.functions, solver.objective, state),
            filter_spec=serialise_model_leaf,
        ),
    )
    manifest = {
        "format": _FUNCTIONAL_CHECKPOINT_FORMAT,
        "state_file": state_path.name,
        "state_sha256": checksum,
        "step": int(state.progress.update_step),
        **_serialize_root_key(state.key),
        "plan_id": plan.plan_id,
        "run_id": state.run_id,
        "target_policy": _target_policy_contract(state),
        "discretization_bundle_id": solver.discretization_bundle.bundle_id,
        "progress": asdict(state.progress),
        "training_seconds": state.training_seconds,
        "resumed_from_step": state.resumed_from_step,
        "update_boundary": True,
    }
    _publish_manifest(destination / "manifest.json", manifest)
    _prune_state_files(destination, state_path.name)
    return destination


def _read_functional_manifest(path: str | Path, /) -> tuple[dict[str, Any], Path]:
    source = Path(path)
    manifest = _read_manifest(source / "manifest.json")
    expected = {
        "format",
        "state_file",
        "state_sha256",
        "step",
        "key_data",
        "key_impl",
        "plan_id",
        "run_id",
        "target_policy",
        "discretization_bundle_id",
        "progress",
        "training_seconds",
        "resumed_from_step",
        "update_boundary",
    }
    if not isinstance(manifest, dict):
        raise TypeError("Functional checkpoint manifest must be an object.")
    missing = expected - set(manifest)
    unknown = set(manifest) - expected
    if missing or unknown:
        raise ValueError(
            "Functional checkpoint fields are not canonical; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}."
        )
    if manifest["format"] != _FUNCTIONAL_CHECKPOINT_FORMAT:
        raise ValueError("File is not a Phydrax functional training checkpoint.")
    if manifest["update_boundary"] is not True:
        raise ValueError("Functional checkpoints must be accepted-update boundaries.")
    state_name = manifest["state_file"]
    if not isinstance(state_name, str) or not state_name:
        raise ValueError("Functional checkpoint state_file must be non-empty.")
    state_path = source / state_name
    _verify_state(state_path, manifest["state_sha256"])
    return manifest, state_path


def load_functional_training_checkpoint(
    path: str | Path,
    solver_like: Any,
    state_like: FunctionalTrainingState,
    plan: FunctionalTrainingPlan,
    /,
) -> FunctionalTrainingCheckpoint:
    """Verify and restore a checkpoint against exact solver and run contracts."""
    if not isinstance(state_like, FunctionalTrainingState):
        raise TypeError("state_like must be a FunctionalTrainingState.")
    if not isinstance(plan, FunctionalTrainingPlan):
        raise TypeError("plan must be a FunctionalTrainingPlan.")
    manifest, state_path = _read_functional_manifest(path)
    if manifest["plan_id"] != plan.plan_id:
        raise ValueError("Functional checkpoint training-plan identity mismatch.")
    if manifest["target_policy"] != _target_policy_contract(state_like):
        raise ValueError("Functional checkpoint target-policy identity mismatch.")
    if (
        manifest["discretization_bundle_id"]
        != solver_like.discretization_bundle.bundle_id
    ):
        raise ValueError("Functional checkpoint discretization identity mismatch.")
    if manifest["run_id"] != state_like.run_id:
        raise ValueError("Functional checkpoint run identity mismatch.")
    functions, objective, restored = eqx.tree_deserialise_leaves(
        state_path,
        (solver_like.functions, solver_like.objective, state_like),
        filter_spec=deserialise_model_leaf,
    )
    progress = TrainingProgress(**manifest["progress"])
    if progress.update_step != int(manifest["step"]):
        raise ValueError("Functional checkpoint progress disagrees with its step.")
    manifest_key = _deserialize_root_key(
        manifest["key_data"],
        manifest["key_impl"],
    )
    if str(jr.key_impl(restored.key)) != str(jr.key_impl(manifest_key)) or not bool(
        jnp.array_equal(jr.key_data(restored.key), jr.key_data(manifest_key))
    ):
        raise ValueError("Functional checkpoint PRNG state disagrees with its manifest.")
    if _target_policy_contract(restored) != manifest["target_policy"]:
        raise ValueError(
            "Functional checkpoint target policy disagrees with its manifest."
        )
    if restored.target_state is not None and int(
        restored.target_state.update_count
    ) != int(progress.update_step):
        raise ValueError("Functional checkpoint target state disagrees with its step.")
    restored = FunctionalTrainingState(
        current_functions=restored.current_functions,
        best_functions=restored.best_functions,
        previous_functions=restored.previous_functions,
        optimizer_state=restored.optimizer_state,
        target_state=restored.target_state,
        key=restored.key,
        pseudo_inverse_steps=restored.pseudo_inverse_steps,
        term_multipliers=restored.term_multipliers,
        previous_gradient=restored.previous_gradient,
        progress=progress,
        run_id=str(manifest["run_id"]),
        training_seconds=float(manifest["training_seconds"]),
        resumed_from_step=int(manifest["resumed_from_step"]),
    )
    return FunctionalTrainingCheckpoint(functions, objective, restored, Path(path))


__all__ = [
    "FunctionalTrainingCheckpoint",
    "load_functional_training_checkpoint",
    "save_functional_training_checkpoint",
]
