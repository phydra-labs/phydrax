#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any, Literal, TypeAlias

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._metric import LorentzianConvention


OrientationSign: TypeAlias = Literal[-1, 1]


def _sign(value: int, name: str, /) -> OrientationSign:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer sign.")
    resolved = int(value)
    if resolved not in (-1, 1):
        raise ValueError(f"{name} must be either -1 or +1.")
    return resolved


class RelativityConvention(StrictModule, NonTrainableState):
    """Complete static sign and orientation identity for relativity calculations.

    ``riemann_sign`` multiplies the convention
    ``R^rho_{ sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu
    Gamma^rho_{mu sigma} + ...``. ``extrinsic_curvature_sign`` multiplies
    ``(1/2) L_n gamma``. The remaining integer fields declare positive
    spacetime volume, time, azimuth, and Fourier phase orientations.
    """

    metric_signature: LorentzianConvention = eqx.field(static=True)
    riemann_sign: OrientationSign = eqx.field(static=True)
    extrinsic_curvature_sign: OrientationSign = eqx.field(static=True)
    spacetime_orientation: OrientationSign = eqx.field(static=True)
    future_time_orientation: OrientationSign = eqx.field(static=True)
    azimuthal_orientation: OrientationSign = eqx.field(static=True)
    fourier_sign: OrientationSign = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)

    def __init__(
        self,
        metric_signature: LorentzianConvention = "mostly_plus",
        riemann_sign: int = 1,
        extrinsic_curvature_sign: int = -1,
        spacetime_orientation: int = 1,
        future_time_orientation: int = 1,
        azimuthal_orientation: int = 1,
        fourier_sign: int = -1,
    ):
        if metric_signature not in ("mostly_plus", "mostly_minus"):
            raise ValueError("metric_signature must be 'mostly_plus' or 'mostly_minus'.")
        riemann = _sign(riemann_sign, "riemann_sign")
        extrinsic = _sign(extrinsic_curvature_sign, "extrinsic_curvature_sign")
        spacetime = _sign(spacetime_orientation, "spacetime_orientation")
        future = _sign(future_time_orientation, "future_time_orientation")
        azimuthal = _sign(azimuthal_orientation, "azimuthal_orientation")
        fourier = _sign(fourier_sign, "fourier_sign")
        self.metric_signature = metric_signature
        self.riemann_sign = riemann
        self.extrinsic_curvature_sign = extrinsic
        self.spacetime_orientation = spacetime
        self.future_time_orientation = future
        self.azimuthal_orientation = azimuthal
        self.fourier_sign = fourier
        self.convention_id = canonical_fingerprint(
            {
                "kind": "relativity-convention",
                "metric_signature": metric_signature,
                "riemann_sign": riemann,
                "extrinsic_curvature_sign": extrinsic,
                "spacetime_orientation": spacetime,
                "future_time_orientation": future,
                "azimuthal_orientation": azimuthal,
                "fourier_sign": fourier,
            }
        )

    @classmethod
    def canonical(cls) -> RelativityConvention:
        """Return the package's declared mostly-plus, future-oriented preset."""
        return cls()

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_signature": self.metric_signature,
            "riemann_sign": self.riemann_sign,
            "extrinsic_curvature_sign": self.extrinsic_curvature_sign,
            "spacetime_orientation": self.spacetime_orientation,
            "future_time_orientation": self.future_time_orientation,
            "azimuthal_orientation": self.azimuthal_orientation,
            "fourier_sign": self.fourier_sign,
            "convention_id": self.convention_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RelativityConvention:
        if not isinstance(payload, Mapping):
            raise TypeError("Relativity convention payload must be a mapping.")
        expected = {
            "metric_signature",
            "riemann_sign",
            "extrinsic_curvature_sign",
            "spacetime_orientation",
            "future_time_orientation",
            "azimuthal_orientation",
            "fourier_sign",
            "convention_id",
        }
        if set(payload) != expected:
            raise ValueError(
                "Relativity convention payload must use the canonical fields."
            )
        metric_signature = payload["metric_signature"]
        if not isinstance(metric_signature, str):
            raise TypeError("metric_signature payload must be a string.")
        convention = cls(
            metric_signature,
            payload["riemann_sign"],
            payload["extrinsic_curvature_sign"],
            payload["spacetime_orientation"],
            payload["future_time_orientation"],
            payload["azimuthal_orientation"],
            payload["fourier_sign"],
        )
        claimed_id = payload["convention_id"]
        if not isinstance(claimed_id, str):
            raise TypeError("convention_id payload must be a string.")
        if claimed_id != convention.convention_id:
            raise ValueError(
                "Relativity convention fingerprint does not match its content."
            )
        return convention


__all__ = ["OrientationSign", "RelativityConvention"]
