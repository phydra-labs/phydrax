#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, IntFlag
from typing import Any

import equinox as eqx
from jaxtyping import Array

from ..._strict import StrictModule
from ...discretization.splatting import ParticleGridSplatState, SplatBalanceEvidence


class PICRunStatus(IntEnum):
    SUCCESS = 0
    INVALID_STATE = 1
    TRANSFER_FAILED = 2
    FIELD_SOLVE_FAILED = 3
    STABILITY_LIMIT_EXCEEDED = 4
    CURRENT_DEPOSITION_FAILED = 5
    MAXWELL_STEP_FAILED = 6
    NONFINITE_STATE = 7


class PICRejectionReason(IntFlag):
    NONE = 0
    ROUTE = 1
    FIELD = 2
    PUSHER = 4
    DISPLACEMENT = 8
    CONTINUITY = 16
    GAUSS = 32
    MAGNETIC = 64
    NONFINITE = 128


class PICParticleState(StrictModule):
    """Fixed-capacity charged-particle kinematics in dD3V coordinates."""

    position: Array
    proper_velocity: Array


class BorisPushResult(StrictModule):
    proper_velocity: Array
    velocity: Array
    maximum_speed: Array
    finite: Array
    subluminal: Array
    successful: Array


class PICTransferState(StrictModule):
    """Instantaneous charge and oriented field routes for one species."""

    charge: ParticleGridSplatState
    electric: tuple[ParticleGridSplatState, ...]
    magnetic: tuple[ParticleGridSplatState, ...]
    transfer_id: str = eqx.field(static=True)


class PICChargeDepositResult(StrictModule):
    content: Array
    density: Array
    cochain: Array
    balance: SplatBalanceEvidence
    successful: Array
    transfer_id: str = eqx.field(static=True)


class PICFieldGatherResult(StrictModule):
    values: Array
    support: Array
    finite: Array
    successful: Array
    transfer_id: str = eqx.field(static=True)


class PICCurrentDepositResult(StrictModule):
    start_charge: PICChargeDepositResult
    end_charge: PICChargeDepositResult
    current: Array
    continuity_residual: Array
    maximum_continuity_defect: Array
    segment_count: Array
    capacity_overflow: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class PICEnergyLedger(StrictModule):
    particle_kinetic: Array
    electric_field: Array
    magnetic_field: Array
    total: Array
    previous_total: Array
    defect: Array


class PICStepEvidence(StrictModule):
    status: Array
    rejection_reason: Array
    charge_balance_defect: Array
    gauss_defect: Array
    magnetic_defect: Array
    continuity_defect: Array
    maximum_displacement_fraction: Array
    pusher_successful: Array
    transfer_successful: Array
    field_successful: Array
    finite: Array
    successful: Array
    diagnostics: Any


__all__ = [
    "BorisPushResult",
    "PICChargeDepositResult",
    "PICCurrentDepositResult",
    "PICEnergyLedger",
    "PICFieldGatherResult",
    "PICParticleState",
    "PICRejectionReason",
    "PICRunStatus",
    "PICStepEvidence",
    "PICTransferState",
]
