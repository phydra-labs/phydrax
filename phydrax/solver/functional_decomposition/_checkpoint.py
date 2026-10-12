#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import equinox as eqx

from ..._model._structure import deserialize_model_leaf, serialize_model_leaf
from ..._training_checkpoint import (
    _open_verified_state,
    _prune_state_files,
    _publish_manifest,
    _publish_state,
    _read_manifest,
)
from ._prepare import PreparedFunctionalDecomposition
from ._solve import FunctionalDecompositionState


_CHECKPOINT_FORMAT = "phydrax-functional-domain-decomposition-checkpoint"


def save_functional_decomposition_checkpoint(
    path: str | Path,
    state: FunctionalDecompositionState,
    prepared: PreparedFunctionalDecomposition,
    /,
) -> Path:
    """Atomically save one accepted completed sweep, including Aitken carry."""
    if not isinstance(state, FunctionalDecompositionState):
        raise TypeError("state must be a FunctionalDecompositionState.")
    if not isinstance(prepared, PreparedFunctionalDecomposition):
        raise TypeError("prepared must be a PreparedFunctionalDecomposition.")
    if state.strategy == "joint":
        raise ValueError(
            "Joint decomposition uses FunctionalTrainingCheckpoint through its underlying FunctionalSolver."
        )
    if len(state.kernel_states) != len(prepared.problem.cover.patches):
        raise ValueError("Checkpoint kernel-state count does not match the cover.")

    destination = Path(path)
    state_path, checksum = _publish_state(
        destination,
        lambda target: eqx.tree_serialise_leaves(
            target,
            state,
            filter_spec=serialize_model_leaf,
        ),
    )
    manifest = {
        "format": _CHECKPOINT_FORMAT,
        "state_file": state_path.name,
        "state_sha256": checksum,
        "prepared_id": prepared.prepared_id,
        "cover_id": prepared.problem.cover.cover_id,
        "plan_id": prepared.plan.plan_id,
        "completed_sweeps": state.completed_sweeps,
        "strategy": state.strategy,
        "has_previous_increment": state.previous_increment is not None,
        "has_omega": state.omega is not None,
        "kernel_checkpoint_ids": list(state.kernel_checkpoint_ids),
        "accepted_sweep_boundary": True,
    }
    _publish_manifest(destination / "manifest.json", manifest)
    _prune_state_files(destination, state_path.name)
    return destination


def load_functional_decomposition_checkpoint(
    path: str | Path,
    prepared: PreparedFunctionalDecomposition,
    state_like: FunctionalDecompositionState,
    /,
) -> FunctionalDecompositionState:
    """Verify and restore a decomposition checkpoint against exact run contracts.

    `state_like` must use the canonical state representation with matching
    numerical leaf shapes and dtypes, kernel identities, and completed sweep.
    For Aitken runs its `previous_increment` and `omega` must have the same
    presence as the saved state; their values are restored from the checkpoint.
    """
    if not isinstance(prepared, PreparedFunctionalDecomposition):
        raise TypeError("prepared must be a PreparedFunctionalDecomposition.")
    if not isinstance(state_like, FunctionalDecompositionState):
        raise TypeError("state_like must be a FunctionalDecompositionState.")
    source = Path(path)
    manifest = _read_manifest(source / "manifest.json")
    expected = {
        "format",
        "state_file",
        "state_sha256",
        "prepared_id",
        "cover_id",
        "plan_id",
        "completed_sweeps",
        "strategy",
        "has_previous_increment",
        "has_omega",
        "kernel_checkpoint_ids",
        "accepted_sweep_boundary",
    }
    if not isinstance(manifest, dict):
        raise TypeError("Decomposition checkpoint manifest must be an object.")
    missing = expected - set(manifest)
    unknown = set(manifest) - expected
    if missing or unknown:
        raise ValueError(
            f"Decomposition checkpoint fields are not canonical; missing={sorted(missing)}, unknown={sorted(unknown)}."
        )
    if manifest["format"] != _CHECKPOINT_FORMAT:
        raise ValueError("File is not a Phydrax decomposition checkpoint.")
    if manifest["accepted_sweep_boundary"] is not True:
        raise ValueError("Decomposition checkpoints must be complete sweep boundaries.")
    if manifest["prepared_id"] != prepared.prepared_id:
        raise ValueError("Decomposition checkpoint prepared identity mismatch.")
    if manifest["cover_id"] != prepared.problem.cover.cover_id:
        raise ValueError("Decomposition checkpoint cover identity mismatch.")
    if manifest["plan_id"] != prepared.plan.plan_id:
        raise ValueError("Decomposition checkpoint plan identity mismatch.")
    if manifest["strategy"] != state_like.strategy:
        raise ValueError("Decomposition checkpoint strategy mismatch.")
    if manifest["kernel_checkpoint_ids"] != list(state_like.kernel_checkpoint_ids):
        raise ValueError(
            "Decomposition checkpoint patch kernel identities do not match state_like."
        )
    if manifest["has_previous_increment"] is not (
        state_like.previous_increment is not None
    ) or manifest["has_omega"] is not (state_like.omega is not None):
        raise ValueError(
            "Decomposition checkpoint Aitken carry structure does not match state_like."
        )
    state_name = manifest["state_file"]
    with _open_verified_state(
        source,
        state_name,
        manifest["state_sha256"],
    ) as state_stream:
        restored = eqx.tree_deserialise_leaves(
            state_stream,
            state_like,
            filter_spec=deserialize_model_leaf,
        )
    if restored.completed_sweeps != int(manifest["completed_sweeps"]):
        raise ValueError("Decomposition checkpoint sweep count is inconsistent.")
    return restored


__all__ = [
    "load_functional_decomposition_checkpoint",
    "save_functional_decomposition_checkpoint",
]
