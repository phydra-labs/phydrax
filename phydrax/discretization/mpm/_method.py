#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._schedule import AbstractExplicitMPMSchedule, USLMPMSchedule
from ._velocity_transfer import (
    AbstractMPMAdvectionPlan,
    AbstractMPMVelocityTransferPlan,
    APICTransferPlan,
    FLIPTransferPlan,
    PICAdvectionPlan,
    TransferredVelocityAdvectionPlan,
)


class MPMResourcePolicy(StrictModule, NonTrainableState):
    maximum_step_workspace_bytes: int = eqx.field(static=True)
    maximum_state_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_step_workspace_bytes: int = 2 * 1024**3,
        maximum_state_bytes: int = 1024**3,
    ):
        workspace = int(maximum_step_workspace_bytes)
        state = int(maximum_state_bytes)
        if workspace <= 0 or state <= 0:
            raise ValueError("MPM resource limits must be positive.")
        self.maximum_step_workspace_bytes = workspace
        self.maximum_state_bytes = state
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mpm-resource-policy",
                "maximum_step_workspace_bytes": workspace,
                "maximum_state_bytes": state,
            }
        )

    def admit(self, *, step_workspace_bytes: int, state_bytes: int) -> None:
        if int(step_workspace_bytes) > self.maximum_step_workspace_bytes:
            raise ValueError("MPM step workspace exceeds its resource policy.")
        if int(state_bytes) > self.maximum_state_bytes:
            raise ValueError("MPM particle state exceeds its resource policy.")


class ExplicitMPMMethodPlan(StrictModule, NonTrainableState):
    """Explicit updated-Lagrangian MPM with independent transfer/advection/schedule."""

    transfer: AbstractMPMVelocityTransferPlan
    advection: AbstractMPMAdvectionPlan
    schedule: AbstractExplicitMPMSchedule
    acoustic_cfl: float = eqx.field(static=True)
    advective_cfl: float = eqx.field(static=True)
    force_cfl: float = eqx.field(static=True)
    mass_tolerance_factor: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: AbstractMPMVelocityTransferPlan | None = None,
        /,
        *,
        advection: AbstractMPMAdvectionPlan | None = None,
        schedule: AbstractExplicitMPMSchedule | None = None,
        acoustic_cfl: float = 0.4,
        advective_cfl: float = 0.4,
        force_cfl: float = 0.25,
        mass_tolerance_factor: float = 32.0,
    ):
        transfer_ = APICTransferPlan() if transfer is None else transfer
        if not isinstance(transfer_, AbstractMPMVelocityTransferPlan):
            raise TypeError("transfer must be AbstractMPMVelocityTransferPlan or None.")
        if advection is None:
            advection_ = (
                PICAdvectionPlan()
                if isinstance(transfer_, FLIPTransferPlan)
                else TransferredVelocityAdvectionPlan()
            )
        else:
            advection_ = advection
        if not isinstance(advection_, AbstractMPMAdvectionPlan):
            raise TypeError("advection must be AbstractMPMAdvectionPlan or None.")
        schedule_ = USLMPMSchedule() if schedule is None else schedule
        if not isinstance(schedule_, AbstractExplicitMPMSchedule):
            raise TypeError("schedule must be AbstractExplicitMPMSchedule or None.")
        values = tuple(
            float(value)
            for value in (
                acoustic_cfl,
                advective_cfl,
                force_cfl,
                mass_tolerance_factor,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("MPM CFL and mass-tolerance factors must be positive.")
        self.transfer = transfer_
        self.advection = advection_
        self.schedule = schedule_
        self.acoustic_cfl = values[0]
        self.advective_cfl = values[1]
        self.force_cfl = values[2]
        self.mass_tolerance_factor = values[3]
        self.method_id = canonical_fingerprint(
            {
                "kind": "explicit-mpm-method",
                "stress_update": schedule_.stress_update,
                "schedule": schedule_.schedule_id,
                "deformation_update": "forward-euler",
                "transfer": transfer_.plan_id,
                "advection": advection_.plan_id,
                "acoustic_cfl": values[0],
                "advective_cfl": values[1],
                "force_cfl": values[2],
                "mass_tolerance_factor": values[3],
            }
        )


__all__ = ["APICTransferPlan", "ExplicitMPMMethodPlan", "MPMResourcePolicy"]
