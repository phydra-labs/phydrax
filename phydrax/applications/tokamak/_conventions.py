#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit axisymmetric tokamak coordinates, signs, and flux normalization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ...units import METER


def _sign(value: int, name: str, /) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) not in (-1, 1)
    ):
        raise ValueError(f"{name} must be exactly -1 or +1.")
    return int(value)


def _text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{name} must be non-empty canonical text.")
    return normalized


class PoloidalFluxNormalization(StrEnum):
    WEBER = "weber"
    WEBER_PER_RADIAN = "weber_per_radian"

    @property
    def two_pi_exponent(self) -> int:
        return int(self is PoloidalFluxNormalization.WEBER)


@dataclass(frozen=True, slots=True)
class TokamakMagneticConvention:
    """Complete signs and normalization needed to interpret axisymmetric fields."""

    name: str
    poloidal_flux_normalization: PoloidalFluxNormalization
    poloidal_field_sign: int
    coordinate_handedness: int
    radial_toroidal_flux_sign: int
    plasma_current_sign: int
    toroidal_field_sign: int
    safety_factor_sign: int
    cocos_index: int | None = None
    convention_id: str = field(init=False)

    def __post_init__(self) -> None:
        name = _text(self.name, "name")
        if not isinstance(self.poloidal_flux_normalization, PoloidalFluxNormalization):
            raise TypeError(
                "poloidal_flux_normalization must be PoloidalFluxNormalization."
            )
        values = tuple(
            _sign(value, label)
            for value, label in (
                (self.poloidal_field_sign, "poloidal_field_sign"),
                (self.coordinate_handedness, "coordinate_handedness"),
                (self.radial_toroidal_flux_sign, "radial_toroidal_flux_sign"),
                (self.plasma_current_sign, "plasma_current_sign"),
                (self.toroidal_field_sign, "toroidal_field_sign"),
                (self.safety_factor_sign, "safety_factor_sign"),
            )
        )
        cocos = self.cocos_index
        if cocos is not None:
            if isinstance(cocos, bool) or not isinstance(cocos, Integral):
                raise TypeError("cocos_index must be an integer or None.")
            cocos = int(cocos)
            if cocos < 1 or cocos > 18:
                raise ValueError(
                    "cocos_index must be in the documented range 1 through 18."
                )
        object.__setattr__(self, "name", name)
        (
            poloidal,
            handedness,
            radial_flux,
            plasma_current,
            toroidal,
            safety,
        ) = values
        object.__setattr__(self, "poloidal_field_sign", poloidal)
        object.__setattr__(self, "coordinate_handedness", handedness)
        object.__setattr__(self, "radial_toroidal_flux_sign", radial_flux)
        object.__setattr__(self, "plasma_current_sign", plasma_current)
        object.__setattr__(self, "toroidal_field_sign", toroidal)
        object.__setattr__(self, "safety_factor_sign", safety)
        object.__setattr__(self, "cocos_index", cocos)
        object.__setattr__(
            self,
            "convention_id",
            canonical_fingerprint(
                {
                    "kind": "tokamak-magnetic-convention",
                    "name": name,
                    "flux_normalization": self.poloidal_flux_normalization.value,
                    "poloidal_field_sign": poloidal,
                    "coordinate_handedness": handedness,
                    "radial_toroidal_flux_sign": radial_flux,
                    "plasma_current_sign": plasma_current,
                    "toroidal_field_sign": toroidal,
                    "safety_factor_sign": safety,
                    "cocos_index": cocos,
                }
            ),
        )

    @classmethod
    def canonical(cls) -> TokamakMagneticConvention:
        return cls(
            "phydrax-r-phi-z",
            PoloidalFluxNormalization.WEBER_PER_RADIAN,
            poloidal_field_sign=-1,
            coordinate_handedness=1,
            radial_toroidal_flux_sign=1,
            plasma_current_sign=1,
            toroidal_field_sign=1,
            safety_factor_sign=1,
        )

    def transform_to(
        self, target: TokamakMagneticConvention, /
    ) -> TokamakConventionTransform:
        if not isinstance(target, TokamakMagneticConvention):
            raise TypeError("target must be TokamakMagneticConvention.")
        flux_factor = (
            (2.0 * math.pi)
            ** (
                target.poloidal_flux_normalization.two_pi_exponent
                - self.poloidal_flux_normalization.two_pi_exponent
            )
            * self.poloidal_field_sign
            / target.poloidal_field_sign
        )
        return TokamakConventionTransform(
            self.convention_id,
            target.convention_id,
            flux_factor,
            self.plasma_current_sign / target.plasma_current_sign,
            self.toroidal_field_sign / target.toroidal_field_sign,
            self.safety_factor_sign / target.safety_factor_sign,
        )


@dataclass(frozen=True, slots=True)
class TokamakConventionTransform:
    source_convention_id: str
    target_convention_id: str
    poloidal_flux_factor: float
    plasma_current_factor: float
    toroidal_field_factor: float
    safety_factor_factor: float
    transform_id: str = field(init=False)

    def __post_init__(self) -> None:
        source = _text(self.source_convention_id, "source_convention_id")
        target = _text(self.target_convention_id, "target_convention_id")
        factors = tuple(
            float(value)
            for value in (
                self.poloidal_flux_factor,
                self.plasma_current_factor,
                self.toroidal_field_factor,
                self.safety_factor_factor,
            )
        )
        if any(not math.isfinite(value) or value == 0.0 for value in factors):
            raise ValueError("Tokamak convention factors must be finite and nonzero.")
        object.__setattr__(self, "source_convention_id", source)
        object.__setattr__(self, "target_convention_id", target)
        (
            flux,
            current,
            toroidal,
            safety,
        ) = factors
        object.__setattr__(self, "poloidal_flux_factor", flux)
        object.__setattr__(self, "plasma_current_factor", current)
        object.__setattr__(self, "toroidal_field_factor", toroidal)
        object.__setattr__(self, "safety_factor_factor", safety)
        object.__setattr__(
            self,
            "transform_id",
            canonical_fingerprint(
                {
                    "kind": "tokamak-convention-transform",
                    "source": source,
                    "target": target,
                    "factors": list(factors),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class AxisymmetricMachineFrame:
    machine_id: str
    coordinate_contract: SpatialCoordinateContract = field(init=False)
    frame_id: str = field(init=False)

    def __post_init__(self) -> None:
        machine = _text(self.machine_id, "machine_id")
        coordinate = SpatialCoordinateContract(
            METER,
            coordinate_system="cylindrical-r-phi-z",
            reference_frame=machine,
        )
        object.__setattr__(self, "machine_id", machine)
        object.__setattr__(self, "coordinate_contract", coordinate)
        object.__setattr__(
            self,
            "frame_id",
            canonical_fingerprint(
                {
                    "kind": "axisymmetric-machine-frame",
                    "machine": machine,
                    "spatial": coordinate.spatial_id,
                }
            ),
        )


__all__ = [
    "AxisymmetricMachineFrame",
    "PoloidalFluxNormalization",
    "TokamakConventionTransform",
    "TokamakMagneticConvention",
]
