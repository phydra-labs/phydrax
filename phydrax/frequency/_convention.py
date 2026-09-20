#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass
from typing import Literal, TypeAlias

from .._fingerprint import canonical_fingerprint


AmplitudeConvention: TypeAlias = Literal["peak", "rms"]
PhasorConvention: TypeAlias = Literal["exp-positive-iwt", "exp-negative-iwt"]


@dataclass(frozen=True, slots=True)
class HarmonicConvention:
    amplitude: AmplitudeConvention = "peak"
    phasor: PhasorConvention = "exp-positive-iwt"

    @property
    def convention_id(self):
        return canonical_fingerprint(
            {
                "kind": "harmonic-convention",
                "amplitude": self.amplitude,
                "phasor": self.phasor,
            }
        )


__all__ = ["AmplitudeConvention", "HarmonicConvention", "PhasorConvention"]
