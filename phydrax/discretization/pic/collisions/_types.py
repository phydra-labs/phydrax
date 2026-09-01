#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array

from ...._strict import StrictModule


class PICCollisionResult(StrictModule):
    candidate_velocity: Array
    accepted_velocity: Array
    collided: Array
    pair_count: Array
    momentum_defect: Array
    energy_defect: Array
    background_momentum_source: Array
    background_energy_source: Array
    maximum_probability: Array
    finite: Array
    stable: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


__all__ = ["PICCollisionResult"]
