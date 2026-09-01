#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array

from ...._strict import StrictModule
from ...particle import ParticlePopulationState
from .._charge_state import PICChargeState
from .._types import PICParticleState


class PICIonizationResult(StrictModule):
    ion_charge: PICChargeState
    ion_particles: PICParticleState
    electron_population: ParticlePopulationState
    electron_particles: PICParticleState
    electron_charge: PICChargeState
    event_mask: Array
    event_count: Array
    charge_defect: Array
    momentum_defect: Array
    energy_defect: Array
    ionization_energy: Array
    capacity_available: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


__all__ = ["PICIonizationResult"]
