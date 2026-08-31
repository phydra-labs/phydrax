#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class CosmologyScaleContract(StrictModule, NonTrainableState):
    """Explicit comoving length, mass, and physical-time scale identity."""

    length_unit: str = eqx.field(static=True)
    mass_unit: str = eqx.field(static=True)
    time_unit: str = eqx.field(static=True)
    length_to_reference: float = eqx.field(static=True)
    mass_to_reference: float = eqx.field(static=True)
    time_to_reference: float = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)

    def __init__(
        self,
        length_unit: str,
        mass_unit: str,
        time_unit: str,
        /,
        *,
        length_to_reference: float = 1.0,
        mass_to_reference: float = 1.0,
        time_to_reference: float = 1.0,
    ):
        length = str(length_unit).strip()
        mass = str(mass_unit).strip()
        time = str(time_unit).strip()
        factors = tuple(
            float(value)
            for value in (length_to_reference, mass_to_reference, time_to_reference)
        )
        if not length or not mass or not time:
            raise ValueError("Cosmology scale units must be non-empty.")
        if any(not np.isfinite(value) or value <= 0.0 for value in factors):
            raise ValueError("Cosmology reference factors must be finite and positive.")
        self.length_unit = length
        self.mass_unit = mass
        self.time_unit = time
        self.length_to_reference, self.mass_to_reference, self.time_to_reference = factors
        self.scale_id = canonical_fingerprint(
            {
                "kind": "cosmology-scale-contract",
                "length_unit": length,
                "mass_unit": mass,
                "time_unit": time,
                "length_to_reference": factors[0],
                "mass_to_reference": factors[1],
                "time_to_reference": factors[2],
            }
        )

    @property
    def hubble_unit(self) -> str:
        return f"1/{self.time_unit}"

    @property
    def wavenumber_unit(self) -> str:
        return f"1/{self.length_unit}"

    @property
    def power_spectrum_unit(self) -> str:
        return f"{self.length_unit}^3"

    @property
    def potential_unit(self) -> str:
        return f"{self.length_unit}^2/{self.time_unit}^2"

    @property
    def acceleration_unit(self) -> str:
        return f"{self.length_unit}/{self.time_unit}^2"

    @property
    def canonical_momentum_unit(self) -> str:
        return f"{self.mass_unit}*{self.length_unit}/{self.time_unit}"

    @property
    def gravitational_constant_unit(self) -> str:
        return f"{self.length_unit}^3/({self.mass_unit}*{self.time_unit}^2)"


CODE_COSMOLOGY_SCALE = CosmologyScaleContract(
    "code_length",
    "code_mass",
    "code_time",
)


__all__ = ["CODE_COSMOLOGY_SCALE", "CosmologyScaleContract"]
