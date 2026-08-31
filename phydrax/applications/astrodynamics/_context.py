#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


AstrodynamicsTimeScale: TypeAlias = Literal["TAI", "TT", "TDB"]


class AstrodynamicsScaleContract(StrictModule, NonTrainableState):
    """Explicit length, mass, and time scale identity for astrodynamics arrays."""

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
        units = tuple(str(value).strip() for value in (length_unit, mass_unit, time_unit))
        factors = tuple(
            float(value)
            for value in (length_to_reference, mass_to_reference, time_to_reference)
        )
        if any(not value for value in units):
            raise ValueError("Astrodynamics unit names must be non-empty.")
        if any(not np.isfinite(value) or value <= 0.0 for value in factors):
            raise ValueError(
                "Astrodynamics reference factors must be finite and positive."
            )
        self.length_unit, self.mass_unit, self.time_unit = units
        (
            self.length_to_reference,
            self.mass_to_reference,
            self.time_to_reference,
        ) = factors
        self.scale_id = canonical_fingerprint(
            {
                "kind": "astrodynamics-scale-contract",
                "units": list(units),
                "reference_factors": list(factors),
            }
        )

    @classmethod
    def si(cls) -> AstrodynamicsScaleContract:
        return cls("m", "kg", "s")

    @property
    def velocity_unit(self) -> str:
        return f"{self.length_unit}/{self.time_unit}"

    @property
    def acceleration_unit(self) -> str:
        return f"{self.length_unit}/{self.time_unit}^2"

    @property
    def gravitational_parameter_unit(self) -> str:
        return f"{self.length_unit}^3/{self.time_unit}^2"


class ReferenceEpoch(StrictModule, NonTrainableState):
    """Two-part Julian date defining the origin of relative solver time."""

    jd1: float = eqx.field(static=True)
    jd2: float = eqx.field(static=True)
    time_scale: AstrodynamicsTimeScale = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(self, jd1: float, jd2: float, time_scale: AstrodynamicsTimeScale, /):
        first = float(jd1)
        second = float(jd2)
        scale = str(time_scale).upper()
        if not np.isfinite(first) or not np.isfinite(second):
            raise ValueError("Reference epoch parts must be finite.")
        if abs(second) >= 1.0:
            raise ValueError("Reference epoch jd2 must have magnitude below one day.")
        if scale not in ("TAI", "TT", "TDB"):
            raise ValueError("Reference epoch scale must be TAI, TT, or TDB.")
        self.jd1 = first
        self.jd2 = second
        self.time_scale = scale  # type: ignore[assignment]
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "astrodynamics-reference-epoch",
                "jd1": first,
                "jd2": second,
                "time_scale": scale,
            }
        )


class AstrodynamicsFrame(StrictModule, NonTrainableState):
    """Reference-frame identity for Cartesian astrodynamics states."""

    origin_id: str = eqx.field(static=True)
    orientation_id: str = eqx.field(static=True)
    pseudo_inertial: bool = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)

    def __init__(
        self,
        origin_id: str,
        orientation_id: str,
        /,
        *,
        pseudo_inertial: bool,
    ):
        origin = str(origin_id).strip()
        orientation = str(orientation_id).strip()
        if not origin or not orientation:
            raise ValueError("Astrodynamics frame identifiers must be non-empty.")
        if not isinstance(pseudo_inertial, bool):
            raise TypeError("pseudo_inertial must be a bool.")
        self.origin_id = origin
        self.orientation_id = orientation
        self.pseudo_inertial = pseudo_inertial
        self.frame_id = canonical_fingerprint(
            {
                "kind": "astrodynamics-frame",
                "origin": origin,
                "orientation": orientation,
                "pseudo_inertial": pseudo_inertial,
            }
        )


class AstrodynamicsContext(StrictModule, NonTrainableState):
    """Static scale, epoch, and frame contract for one astrodynamics problem."""

    scale: AstrodynamicsScaleContract
    epoch: ReferenceEpoch
    frame: AstrodynamicsFrame
    context_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: AstrodynamicsScaleContract,
        epoch: ReferenceEpoch,
        frame: AstrodynamicsFrame,
        /,
    ):
        if not isinstance(scale, AstrodynamicsScaleContract):
            raise TypeError("scale must be an AstrodynamicsScaleContract.")
        if not isinstance(epoch, ReferenceEpoch):
            raise TypeError("epoch must be a ReferenceEpoch.")
        if not isinstance(frame, AstrodynamicsFrame):
            raise TypeError("frame must be an AstrodynamicsFrame.")
        self.scale = scale
        self.epoch = epoch
        self.frame = frame
        self.context_id = canonical_fingerprint(
            {
                "kind": "astrodynamics-context",
                "scale": scale.scale_id,
                "epoch": epoch.epoch_id,
                "frame": frame.frame_id,
            }
        )

    def require_compatible(self, other: AstrodynamicsContext, /) -> None:
        if not isinstance(other, AstrodynamicsContext):
            raise TypeError("other must be an AstrodynamicsContext.")
        if self.context_id != other.context_id:
            raise ValueError("Astrodynamics contexts are incompatible.")


__all__ = [
    "AstrodynamicsContext",
    "AstrodynamicsFrame",
    "AstrodynamicsScaleContract",
    "AstrodynamicsTimeScale",
    "ReferenceEpoch",
]
