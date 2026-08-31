#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState


class AbstractExplicitMPMSchedule(StrictModule, NonTrainableState):
    """Static phase ordering for one explicit material-point attempt."""

    schedule_code: AbstractAttribute[int]
    stress_update: AbstractAttribute[str]
    second_momentum_extrapolation: AbstractAttribute[bool]
    schedule_id: AbstractAttribute[str]

    @property
    @abc.abstractmethod
    def common_name(self) -> str:
        raise NotImplementedError


class USLMPMSchedule(AbstractExplicitMPMSchedule):
    """Stress-last schedule without a second momentum extrapolation (USL-minus)."""

    schedule_code: int = eqx.field(static=True)
    stress_update: str = eqx.field(static=True)
    second_momentum_extrapolation: bool = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self):
        self.schedule_code = 0
        self.stress_update = "last"
        self.second_momentum_extrapolation = False
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "mpm-explicit-schedule",
                "common_name": "usl-minus",
                "stress_update": "last",
                "second_momentum_extrapolation": False,
            }
        )

    @property
    def common_name(self) -> str:
        return "usl-minus"


class USFMPMSchedule(AbstractExplicitMPMSchedule):
    """Update stress from the pre-force grid velocity before force integration."""

    schedule_code: int = eqx.field(static=True)
    stress_update: str = eqx.field(static=True)
    second_momentum_extrapolation: bool = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self):
        self.schedule_code = 1
        self.stress_update = "first"
        self.second_momentum_extrapolation = False
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "mpm-explicit-schedule",
                "common_name": "usf",
                "stress_update": "first",
                "second_momentum_extrapolation": False,
            }
        )

    @property
    def common_name(self) -> str:
        return "usf"


class MUSLMPMSchedule(AbstractExplicitMPMSchedule):
    """Classical stress-last schedule with pre-advection translational retransfer."""

    schedule_code: int = eqx.field(static=True)
    stress_update: str = eqx.field(static=True)
    second_momentum_extrapolation: bool = eqx.field(static=True)
    second_route_time: str = eqx.field(static=True)
    second_transfer_mode: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self):
        self.schedule_code = 2
        self.stress_update = "last"
        self.second_momentum_extrapolation = True
        self.second_route_time = "pre-advection"
        self.second_transfer_mode = "translational-momentum"
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "mpm-explicit-schedule",
                "common_name": "musl",
                "stress_update": "last",
                "second_momentum_extrapolation": True,
                "second_route_time": self.second_route_time,
                "second_transfer_mode": self.second_transfer_mode,
            }
        )

    @property
    def common_name(self) -> str:
        return "musl"


__all__ = [
    "AbstractExplicitMPMSchedule",
    "MUSLMPMSchedule",
    "USFMPMSchedule",
    "USLMPMSchedule",
]
