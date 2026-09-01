#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..discretization.mpm import (
    MPMCommercialFailure,
    MPMOperationalStatus,
    MPMRuntimeState,
    PreparedMPMDynamics,
)
from ..equations import MaterialPointArguments
from ._material_point_checkpoint import MPMCheckpointPlan
from ._material_point_output import MPMOutputPlan


class MPMOperationalResult:
    def __init__(
        self,
        state,
        numerical_result,
        *,
        status,
        failure,
        generation,
        output_complete,
        elapsed_seconds,
    ):
        self.state = state
        self.numerical_result = numerical_result
        self.status = MPMOperationalStatus(status)
        self.failure = MPMCommercialFailure(failure)
        self.generation = int(generation)
        self.output_complete = bool(output_complete)
        self.elapsed_seconds = float(elapsed_seconds)


class MPMRunSupervisor:
    """Host lifecycle; numerical retries remain owned by adaptive rollout."""

    def __init__(
        self,
        dynamics: PreparedMPMDynamics,
        initial_state: MPMRuntimeState,
        arguments: MaterialPointArguments,
        /,
        *,
        checkpoint_plan: MPMCheckpointPlan | None = None,
        checkpoint_directory: str | Path | None = None,
        checkpoint_interval: int = 1,
        output_plan: MPMOutputPlan | None = None,
    ):
        if not isinstance(dynamics, PreparedMPMDynamics):
            raise TypeError("dynamics must be PreparedMPMDynamics.")
        if not isinstance(initial_state, MPMRuntimeState):
            raise TypeError("initial_state must be MPMRuntimeState.")
        if not isinstance(arguments, MaterialPointArguments):
            raise TypeError("arguments must be MaterialPointArguments.")
        interval = int(checkpoint_interval)
        if interval <= 0:
            raise ValueError("checkpoint_interval must be positive.")
        if checkpoint_plan is not None and checkpoint_directory is None:
            raise ValueError("Checkpoint plan requires checkpoint_directory.")
        self.dynamics = dynamics
        self.state = initial_state
        self.arguments = arguments
        self.checkpoint_plan = checkpoint_plan
        self.checkpoint_directory = (
            None if checkpoint_directory is None else Path(checkpoint_directory)
        )
        self.checkpoint_interval = interval
        self.output_plan = output_plan
        self.status = MPMOperationalStatus.PREPARED
        self.failure = MPMCommercialFailure.NONE
        self.generation = 0
        self.events = []
        self.metrics = {
            "attempted_steps": 0,
            "accepted_steps": int(np.asarray(initial_state.accepted_step)),
            "rejected_steps": 0,
            "checkpoint_generations": 0,
            "output_steps": 0,
            "elapsed_seconds": 0.0,
            "last_status": int(np.asarray(initial_state.last_status)),
            "last_rejection_reasons": 0,
            "maximum_active_nodes": 0,
            "maximum_valid_routes": 0,
            "maximum_apic_condition": 0.0,
        }

    def _event(self, kind, **payload):
        self.events.append(
            {
                "sequence": len(self.events),
                "kind": str(kind),
                "accepted_step": int(np.asarray(self.state.accepted_step)),
                "time_hex": float(np.asarray(self.state.time)).hex(),
                **payload,
            }
        )

    def advance(self, step_size: Any):
        if self.status in (
            MPMOperationalStatus.FAILED,
            MPMOperationalStatus.QUARANTINED,
            MPMOperationalStatus.RELEASED,
        ):
            raise RuntimeError(f"Supervisor cannot advance from {self.status.name}.")
        self.status = MPMOperationalStatus.RUNNING
        started = perf_counter()
        result = self.dynamics.step_detailed(self.state, step_size, self.arguments)
        elapsed = perf_counter() - started
        self.metrics["attempted_steps"] += 1
        self.metrics["elapsed_seconds"] += elapsed
        self.metrics["last_status"] = int(np.asarray(result.accepted_state.last_status))
        self.metrics["last_rejection_reasons"] = int(np.asarray(result.rejection_reasons))
        self.metrics["maximum_active_nodes"] = max(
            self.metrics["maximum_active_nodes"],
            int(np.asarray(result.diagnostics.transfer.active_grid_nodes)),
        )
        self.metrics["maximum_valid_routes"] = max(
            self.metrics["maximum_valid_routes"],
            int(np.asarray(result.diagnostics.transfer.valid_routes)),
        )
        self.metrics["maximum_apic_condition"] = max(
            self.metrics["maximum_apic_condition"],
            float(np.asarray(result.diagnostics.transfer.maximum_apic_condition)),
        )
        if not bool(np.asarray(result.successful)):
            self.metrics["rejected_steps"] += 1
            self.state = result.accepted_state
            self._event(
                "numerical-rejection",
                status=int(np.asarray(result.accepted_state.last_status)),
                reasons=int(np.asarray(result.rejection_reasons)),
                suggested_step=float(np.asarray(result.suggested_step)).hex(),
            )
            return MPMOperationalResult(
                self.state,
                result,
                status=self.status,
                failure=MPMCommercialFailure.NONE,
                generation=self.generation,
                output_complete=True,
                elapsed_seconds=elapsed,
            )
        self.state = result.accepted_state
        self.metrics["accepted_steps"] = int(np.asarray(self.state.accepted_step))
        output_complete = True
        try:
            if self.output_plan is not None:
                self.output_plan.append(self.state)
                self.metrics["output_steps"] += 1
            if (
                self.checkpoint_plan is not None
                and int(np.asarray(self.state.accepted_step)) % self.checkpoint_interval
                == 0
            ):
                self.status = MPMOperationalStatus.CHECKPOINTING
                self.generation += 1
                self.checkpoint_plan.write_generation(
                    self.checkpoint_directory,
                    self.state,
                    generation=self.generation,
                )
                self.metrics["checkpoint_generations"] += 1
                self.status = MPMOperationalStatus.RUNNING
        except (OSError, ValueError, RuntimeError, ImportError, BufferError) as error:
            output_complete = False
            self.status = MPMOperationalStatus.FAILED
            self.failure = (
                MPMCommercialFailure.CHECKPOINT_INTEGRITY_FAILED
                if self.checkpoint_plan is not None
                else MPMCommercialFailure.OUTPUT_INCOMPLETE
            )
            self._event(
                "operational-failure",
                failure=int(self.failure),
                error_type=f"{type(error).__module__}.{type(error).__qualname__}",
                error=str(error),
            )
        else:
            self._event("accepted-step", generation=self.generation)
        return MPMOperationalResult(
            self.state,
            result,
            status=self.status,
            failure=self.failure,
            generation=self.generation,
            output_complete=output_complete,
            elapsed_seconds=elapsed,
        )

    def recover(self):
        if self.checkpoint_plan is None or self.checkpoint_directory is None:
            raise RuntimeError("Supervisor has no checkpoint recovery plan.")
        self.status = MPMOperationalStatus.RECOVERING
        state, manifest = self.checkpoint_plan.read_current(self.checkpoint_directory)
        self.state = state
        self.generation = int(manifest["generation"])
        self.status = MPMOperationalStatus.PREPARED
        self.failure = MPMCommercialFailure.NONE
        self._event("recovered", generation=self.generation)
        return state

    def complete(self):
        if self.status == MPMOperationalStatus.FAILED:
            raise RuntimeError("Failed MPM run cannot be marked complete.")
        self.status = MPMOperationalStatus.COMPLETED
        self._event("completed")

    def quarantine(self, reason: str):
        reason_ = str(reason)
        if not reason_:
            raise ValueError("Quarantine reason must be non-empty.")
        self.status = MPMOperationalStatus.QUARANTINED
        self._event("quarantined", reason=reason_)

    def release(self, release_bundle_id: str):
        identifier = str(release_bundle_id)
        if self.status != MPMOperationalStatus.COMPLETED or not identifier:
            raise RuntimeError("Only completed runs with release evidence may release.")
        self.status = MPMOperationalStatus.RELEASED
        self._event("released", release_bundle_id=identifier)

    def snapshot(self):
        return {
            "status": self.status.name,
            "failure": self.failure.name,
            "generation": self.generation,
            "metrics": dict(self.metrics),
            "events": tuple(dict(value) for value in self.events),
        }


__all__ = ["MPMOperationalResult", "MPMRunSupervisor"]
