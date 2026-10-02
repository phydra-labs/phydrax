#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.flip import MultiphaseFLIPTransferResult
from ..typing import checked
from ._mac_variable_density import (
    MACVariableDensityProjectionPlan,
    MACVariableDensityProjectionResult,
)


class MACMultiphaseProjectionResult(StrictModule):
    projection: MACVariableDensityProjectionResult
    transfer: MultiphaseFLIPTransferResult
    successful: object
    plan_id: str = eqx.field(static=True)


class MACMultiphaseProjectionPlan(StrictModule, NonTrainableState):
    projection: MACVariableDensityProjectionPlan
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(self, projection: MACVariableDensityProjectionPlan, /) -> None:
        self.projection = projection
        self.plan_id = canonical_fingerprint(
            {"kind": "mac-multiphase-projection", "projection": projection.plan_id}
        )

    @checked
    def project(
        self,
        transfer: MultiphaseFLIPTransferResult,
        step_size: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
    ) -> MACMultiphaseProjectionResult:
        result = self.projection.project(
            transfer.face_momentum,
            transfer.face_inverse_density,
            step_size,
            pressure=pressure,
        )
        return MACMultiphaseProjectionResult(
            result,
            transfer,
            transfer.successful & result.successful,
            self.plan_id,
        )


__all__ = ["MACMultiphaseProjectionPlan", "MACMultiphaseProjectionResult"]
