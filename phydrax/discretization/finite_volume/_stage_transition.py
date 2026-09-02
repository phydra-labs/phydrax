#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._unstructured_remap import UnstructuredConservativeRemapPlan


class FiniteVolumeStageEpochTransfer(StrictModule):
    step_start_content: Array
    stage_content: Array
    accepted_ledger_content: Array
    projection_defect: Array
    accepted: Array
    transition_id: str = eqx.field(static=True)


class FiniteVolumeStageEpochTransition(StrictModule, NonTrainableState):
    """Physical payload for one DCD-scheduled transition after SSPRK stage 1/2."""

    source_dynamics_id: str = eqx.field(static=True)
    successor_dynamics: Any
    successor_dynamics_id: str = eqx.field(static=True)
    remap: UnstructuredConservativeRemapPlan
    stage_index: int = eqx.field(static=True)
    event_id: str = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_dynamics_id: str,
        successor_dynamics: Any,
        successor_dynamics_id: str,
        remap: UnstructuredConservativeRemapPlan,
        stage_index: int,
        event_id: str,
        /,
    ):
        source = str(source_dynamics_id)
        successor = str(successor_dynamics_id)
        event = str(event_id)
        stage = int(stage_index)
        if not source or not successor or not event:
            raise ValueError("Stage transition identities must be non-empty.")
        if not isinstance(remap, UnstructuredConservativeRemapPlan):
            raise TypeError("Stage epoch transition requires a conservative remap.")
        if stage not in (1, 2):
            raise ValueError("SSPRK(3,3) internal transition stage must be 1 or 2.")
        if not remap.require_complete or not bool(remap.report.coverage_complete):
            raise ValueError("Stage epoch transition requires complete remap coverage.")
        self.source_dynamics_id = source
        self.successor_dynamics = successor_dynamics
        self.successor_dynamics_id = successor
        self.remap = remap
        self.stage_index = stage
        self.event_id = event
        self.transition_id = canonical_fingerprint(
            {
                "kind": "finite-volume-stage-epoch-transition",
                "source": source,
                "successor": successor,
                "remap": remap.plan_id,
                "stage": stage,
                "event": event,
            }
        )

    def transfer(
        self,
        step_start_content: ArrayLike,
        stage_content: ArrayLike,
        accepted_ledger_content: ArrayLike,
        /,
    ) -> FiniteVolumeStageEpochTransfer:
        start = jnp.asarray(step_start_content)
        stage = jnp.asarray(stage_content)
        ledger = jnp.asarray(accepted_ledger_content)
        if start.shape != stage.shape or start.shape[0] != self.remap.source_volumes.size:
            raise ValueError("Live Shu--Osher registers must match source cell capacity.")
        if ledger.shape[0] != self.remap.source_volumes.size:
            raise ValueError("Accepted ledger accumulator must match source capacity.")
        transferred_start = self.remap.apply_content(start)
        transferred_stage = self.remap.apply_content(stage)
        transferred_ledger = self.remap.apply_content(ledger)
        defects = jnp.stack(
            (
                jnp.max(
                    jnp.abs(
                        self.remap.conservation_defect_content(start, transferred_start)
                    )
                ),
                jnp.max(
                    jnp.abs(
                        self.remap.conservation_defect_content(stage, transferred_stage)
                    )
                ),
                jnp.max(
                    jnp.abs(
                        self.remap.conservation_defect_content(ledger, transferred_ledger)
                    )
                ),
            )
        )
        accepted = (
            jnp.all(jnp.isfinite(transferred_start))
            & jnp.all(jnp.isfinite(transferred_stage))
            & jnp.all(jnp.isfinite(transferred_ledger))
        )
        return FiniteVolumeStageEpochTransfer(
            transferred_start,
            transferred_stage,
            transferred_ledger,
            defects,
            accepted,
            self.transition_id,
        )


__all__ = ["FiniteVolumeStageEpochTransfer", "FiniteVolumeStageEpochTransition"]
