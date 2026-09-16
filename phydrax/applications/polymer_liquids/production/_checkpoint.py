#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ...._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


_FORMAT = "phydrax-polymer-production-checkpoint"


class CompositePolymerCheckpointPlan(StrictModule, NonTrainableState):
    component_names: tuple[str, ...] = eqx.field(static=True)
    component_ids: tuple[str, ...] = eqx.field(static=True)
    regime_id: str = eqx.field(static=True)
    scope_id: str | None = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        components: Mapping[str, str],
        regime_id: str,
        /,
        *,
        scope_id: str | None = None,
    ):
        normalized = tuple(
            sorted(
                (str(name), str(identifier)) for name, identifier in components.items()
            )
        )
        if not normalized or any(
            not name
            or name != name.strip()
            or not identifier
            or identifier != identifier.strip()
            for name, identifier in normalized
        ):
            raise ValueError(
                "Checkpoint component names and IDs must be canonical and nonempty."
            )
        if len({name for name, _ in normalized}) != len(normalized):
            raise ValueError("Checkpoint component names must be unique.")
        regime = str(regime_id)
        scope = None if scope_id is None else str(scope_id)
        if (
            not regime
            or regime != regime.strip()
            or (scope is not None and (not scope or scope != scope.strip()))
        ):
            raise ValueError("Checkpoint regime and scope identities must be canonical.")
        self.component_names = tuple(name for name, _ in normalized)
        self.component_ids = tuple(identifier for _, identifier in normalized)
        self.regime_id = regime
        self.scope_id = scope
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "composite-polymer-checkpoint-plan",
                "components": normalized,
                "regime": regime,
                "scope": scope,
            }
        )

    def state(
        self,
        components: Mapping[str, Any],
        /,
        *,
        component_ids: Mapping[str, str],
        step_index: Array | int = 0,
        accepted_steps: Array | int = 0,
        successful: Array | bool = True,
    ) -> CompositePolymerState:
        if set(components) != set(self.component_names) or set(component_ids) != set(
            self.component_names
        ):
            raise ValueError(
                "Composite state components and runtime IDs must match the checkpoint plan."
            )
        runtime_ids = tuple(str(component_ids[name]) for name in self.component_names)
        if runtime_ids != self.component_ids:
            raise ValueError("Composite component runtime IDs do not match the plan.")
        values = tuple(components[name] for name in self.component_names)
        return CompositePolymerState(
            values,
            jnp.asarray(step_index, dtype=jnp.int32),
            jnp.asarray(accepted_steps, dtype=jnp.int32),
            jnp.asarray(successful),
            self.component_names,
            self.component_ids,
            self.checkpoint_id,
        )


class CompositePolymerState(StrictModule):
    components: tuple[Any, ...]
    step_index: Array
    accepted_steps: Array
    successful: Array
    component_names: tuple[str, ...] = eqx.field(static=True)
    component_ids: tuple[str, ...] = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def component(self, name: str, /) -> Any:
        identifier = str(name)
        if identifier not in self.component_names:
            raise KeyError(identifier)
        return self.components[self.component_names.index(identifier)]


class CompositePolymerCheckpoint(StrictModule):
    state: CompositePolymerState
    payload_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)


class CompositeReplayEvidence(StrictModule, NonTrainableState):
    left_fingerprint: str = eqx.field(static=True)
    right_fingerprint: str = eqx.field(static=True)
    identical: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def _validate_state(
    plan: CompositePolymerCheckpointPlan, state: CompositePolymerState, /
) -> None:
    if not isinstance(plan, CompositePolymerCheckpointPlan) or not isinstance(
        state, CompositePolymerState
    ):
        raise TypeError("plan and state must be composite polymer checkpoint objects.")
    if (
        state.checkpoint_id != plan.checkpoint_id
        or state.component_names != plan.component_names
        or state.component_ids != plan.component_ids
        or len(state.components) != len(plan.component_ids)
    ):
        raise ValueError("Composite state does not belong to this checkpoint plan.")


def write_composite_polymer_checkpoint(
    path: str | Path,
    plan: CompositePolymerCheckpointPlan,
    state: CompositePolymerState,
    /,
) -> CompositePolymerCheckpoint:
    _validate_state(plan, state)
    arrays: dict[str, object] = {}
    specification = pack_array_tree("runtime", state, arrays)
    payload_id = canonical_fingerprint(
        {
            "kind": "composite-polymer-checkpoint-payload",
            "checkpoint": plan.checkpoint_id,
            "step": int(state.step_index),
            "accepted_steps": int(state.accepted_steps),
            "state": specification,
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    manifest = {
        "format": _FORMAT,
        "kind": "polymer-production-runtime",
        "checkpoint_id": plan.checkpoint_id,
        "regime_id": plan.regime_id,
        "component_names": list(plan.component_names),
        "component_ids": list(plan.component_ids),
        "state": specification,
        "payload_id": payload_id,
        **({} if plan.scope_id is None else {"scope_id": plan.scope_id}),
    }
    write_array_archive(path, manifest=manifest, arrays=arrays)
    return CompositePolymerCheckpoint(state, payload_id, plan.checkpoint_id)


def read_composite_polymer_checkpoint(
    path: str | Path,
    plan: CompositePolymerCheckpointPlan,
    template: CompositePolymerState,
    /,
) -> CompositePolymerCheckpoint:
    _validate_state(plan, template)
    manifest, arrays = read_array_archive(path)
    expected = {
        "format",
        "kind",
        "checkpoint_id",
        "regime_id",
        "component_names",
        "component_ids",
        "state",
        "payload_id",
        "arrays",
    }
    if plan.scope_id is not None:
        expected.add("scope_id")
    if set(manifest) != expected:
        raise ValueError(
            "Composite checkpoint manifest is not the canonical current format."
        )
    identities = {
        "format": _FORMAT,
        "kind": "polymer-production-runtime",
        "checkpoint_id": plan.checkpoint_id,
        "regime_id": plan.regime_id,
        "component_names": list(plan.component_names),
        "component_ids": list(plan.component_ids),
    }
    if plan.scope_id is not None:
        identities["scope_id"] = plan.scope_id
    for name, value in identities.items():
        if manifest[name] != value:
            raise ValueError(f"Composite checkpoint {name} is incompatible.")
    state = unpack_array_tree(manifest["state"], arrays, template)
    _validate_state(plan, state)
    specification_arrays: dict[str, object] = {}
    specification = pack_array_tree("runtime", state, specification_arrays)
    payload_id = canonical_fingerprint(
        {
            "kind": "composite-polymer-checkpoint-payload",
            "checkpoint": plan.checkpoint_id,
            "step": int(state.step_index),
            "accepted_steps": int(state.accepted_steps),
            "state": specification,
            "arrays": array_tree_fingerprint(specification_arrays),
        }
    )
    if payload_id != manifest["payload_id"]:
        raise ValueError("Composite checkpoint payload fingerprint does not match.")
    return CompositePolymerCheckpoint(state, payload_id, plan.checkpoint_id)


def compare_composite_polymer_replay(
    left: CompositePolymerState, right: CompositePolymerState, /
) -> CompositeReplayEvidence:
    if not isinstance(left, CompositePolymerState) or not isinstance(
        right, CompositePolymerState
    ):
        raise TypeError("Replay comparison requires composite polymer states.")
    if left.checkpoint_id != right.checkpoint_id:
        raise ValueError("Replay states belong to different checkpoint plans.")

    def fingerprint(state: CompositePolymerState) -> str:
        arrays: dict[str, object] = {}
        specification = pack_array_tree("runtime", state, arrays)
        return canonical_fingerprint(
            {
                "kind": "composite-polymer-replay-state",
                "state": specification,
                "arrays": array_tree_fingerprint(arrays),
            }
        )

    left_id = fingerprint(left)
    right_id = fingerprint(right)
    identical = left_id == right_id
    evidence_id = canonical_fingerprint(
        {
            "kind": "composite-polymer-replay-evidence",
            "left": left_id,
            "right": right_id,
            "identical": identical,
        }
    )
    return CompositeReplayEvidence(left_id, right_id, identical, evidence_id)


__all__ = [
    "CompositePolymerCheckpoint",
    "CompositePolymerCheckpointPlan",
    "CompositePolymerState",
    "CompositeReplayEvidence",
    "compare_composite_polymer_replay",
    "read_composite_polymer_checkpoint",
    "write_composite_polymer_checkpoint",
]
