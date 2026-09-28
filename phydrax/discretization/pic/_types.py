#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, IntFlag
from typing import Any

import equinox as eqx
from jax import Array

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
    PROCESS = 256
    RADIATION_OWNERSHIP = 512
    NUMERICAL_CHERENKOV = 1024
    MIGRATION = 2048


class PICParticleState(StrictModule):
    """Fixed-capacity charged-particle kinematics in dD3V coordinates."""

    position: Array
    proper_velocity: Array


class RelativisticPushResult(StrictModule):
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
    """Charge-conserving current of one step with its endpoint charges.

    ``deposited_end`` is the path head actually deposited: the requested head,
    or its exit point for paths leaving a nonperiodic face, flagged per particle
    by ``boundary_exit``. ``end_charge`` is deposited at ``deposited_end``.
    ``continuity_scale`` is the unsigned charge-rate magnitude
    ``max(|Δρ| + 2ρ_|q|)/Δt`` that ``maximum_continuity_defect`` is certified
    against: the residual is a difference of charges of that size, so its
    roundoff floor is relative to it.
    """

    start_charge: PICChargeDepositResult
    end_charge: PICChargeDepositResult
    current: Array
    continuity_residual: Array
    maximum_continuity_defect: Array
    continuity_scale: Array
    segment_count: Array
    capacity_overflow: Array
    deposited_end: Array
    boundary_exit: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class PICEnergyLedger(StrictModule):
    """Per-step energy ledger.

    ``radiated`` is the energy particles handed this step to radiation the field
    does not resolve (subgrid radiation reaction, net photon emission of
    strong-field QED), ``created_rest_energy`` the rest energy of particles
    created from that radiation (pair creation), and ``field_exchange`` the
    energy the field supplied to close process kinematics (the ``O(m²c⁴/ε)``
    defect of collinear QED events). ``material`` is the energy the medium
    stores (polarization, magnetization, plasma current) and ``dissipated`` the
    energy the field lost this step to conduction, material damping, impedance
    boundaries, and absorbing layers; both are ``None`` when the field solver
    does not account them. ``exited`` is the kinetic energy particles carried
    out through absorbing particle boundaries. ``total`` includes ``material``
    and
    ``defect = total + radiated + created_rest_energy + dissipated + exited −
    field_exchange − previous_total``.
    """

    particle_kinetic: Array
    electric_field: Array
    magnetic_field: Array
    radiated: Array
    total: Array
    previous_total: Array
    defect: Array
    created_rest_energy: Array
    field_exchange: Array
    material: Array | None
    dissipated: Array | None
    exited: Array


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
    "PICChargeDepositResult",
    "PICCurrentDepositResult",
    "PICEnergyLedger",
    "PICFieldGatherResult",
    "PICParticleState",
    "PICRejectionReason",
    "PICRunStatus",
    "PICStepEvidence",
    "PICTransferState",
    "RelativisticPushResult",
]
