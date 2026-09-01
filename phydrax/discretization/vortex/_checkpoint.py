#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class VortexCheckpointRestore(StrictModule):
    state: Any
    event_journal: Any
    rng_state: Any
    accepted_times: Any
    metadata: Mapping[str, Any] = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)


class VortexCheckpointPlan(StrictModule, NonTrainableState):
    state_kind: str = eqx.field(static=True)
    plan_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, state_kind: str, plan_ids: tuple[str, ...], /):
        kind = str(state_kind)
        identifiers = tuple(str(value) for value in plan_ids)
        if not kind or not identifiers or any(not value for value in identifiers):
            raise ValueError("Vortex checkpoint requires state kind and plan IDs.")
        self.state_kind, self.plan_ids = kind, identifiers
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vortex-checkpoint-plan",
                "state_kind": kind,
                "plan_ids": identifiers,
            }
        )

    def write(
        self,
        path: str | Path,
        state: Any,
        event_journal: Any,
        rng_state: Any,
        accepted_times: ArrayLike,
        /,
        *,
        source_lineage_id: str,
        backend_ids: tuple[str, ...],
        epoch_index: int,
    ) -> Path:
        times = jnp.asarray(accepted_times)
        if (
            times.ndim != 1
            or not str(source_lineage_id)
            or any(not str(value) for value in backend_ids)
        ):
            raise ValueError("Checkpoint times, lineage, or backend IDs are invalid.")
        arrays: dict[str, object] = {}
        state_spec = pack_array_tree("state", state, arrays)
        journal_spec = pack_array_tree("journal", event_journal, arrays)
        rng_array = jnp.asarray(rng_state)
        rng_typed = jax.dtypes.issubdtype(
            rng_array.dtype,
            jax.dtypes.prng_key,
        )
        rng_payload = jax.random.key_data(rng_array) if rng_typed else rng_array
        rng_spec = pack_array_tree("rng", rng_payload, arrays)
        time_spec = pack_array_tree("accepted-times", times, arrays)
        checkpoint_id = canonical_fingerprint(
            {
                "kind": "vortex-checkpoint",
                "plan": self.plan_id,
                "source_lineage": str(source_lineage_id),
                "backend_ids": tuple(str(value) for value in backend_ids),
                "epoch_index": int(epoch_index),
                "accepted_count": int(times.size),
            }
        )
        manifest = {
            "kind": "phydrax-vortex-checkpoint",
            "checkpoint_id": checkpoint_id,
            "checkpoint_plan_id": self.plan_id,
            "state_kind": self.state_kind,
            "plan_ids": list(self.plan_ids),
            "source_lineage_id": str(source_lineage_id),
            "backend_ids": [str(value) for value in backend_ids],
            "epoch_index": int(epoch_index),
            "state": state_spec,
            "journal": journal_spec,
            "rng": rng_spec,
            "rng_typed": bool(rng_typed),
            "accepted_times": time_spec,
        }
        return write_array_archive(path, manifest=manifest, arrays=arrays)

    def restore(
        self,
        path: str | Path,
        state_template: Any,
        journal_template: Any,
        rng_template: Any,
        accepted_times_template: Any,
        /,
    ) -> VortexCheckpointRestore:
        manifest, arrays = read_array_archive(path)
        if (
            manifest.get("kind") != "phydrax-vortex-checkpoint"
            or manifest.get("checkpoint_plan_id") != self.plan_id
            or manifest.get("state_kind") != self.state_kind
            or tuple(manifest.get("plan_ids", ())) != self.plan_ids
        ):
            raise ValueError(
                "Vortex checkpoint metadata does not match the runtime plan."
            )
        state = unpack_array_tree(manifest["state"], arrays, state_template)
        journal = unpack_array_tree(manifest["journal"], arrays, journal_template)
        if bool(manifest.get("rng_typed", False)):
            rng_data = unpack_array_tree(
                manifest["rng"],
                arrays,
                jax.random.key_data(rng_template),
            )
            rng = jax.random.wrap_key_data(rng_data)
        else:
            rng = unpack_array_tree(
                manifest["rng"],
                arrays,
                rng_template,
            )
        times = unpack_array_tree(
            manifest["accepted_times"], arrays, accepted_times_template
        )
        return VortexCheckpointRestore(
            state, journal, rng, times, manifest, str(manifest["checkpoint_id"])
        )


__all__ = ["VortexCheckpointPlan", "VortexCheckpointRestore"]
