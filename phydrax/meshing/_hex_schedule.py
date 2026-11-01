# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Canonical schedules for native integer-grid hexahedral routes."""

from __future__ import annotations

from typing import final, Literal

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar
from ..typing import parse


NativeHexGridRoute = Literal["balanced_grid", "frame_grid"]


@final
class NativeHexGridSchedule(StrictModule, NonTrainableState):
    """Numerical quality schedule for integer-grid hexahedral routes."""

    route: NativeHexGridRoute = eqx.field(static=True)
    minimum_scaled_jacobian: float = eqx.field(static=True)
    minimum_mean_ratio: float = eqx.field(static=True)
    maximum_aspect_ratio: float = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        route: NativeHexGridRoute,
        /,
        *,
        minimum_scaled_jacobian: float = 0.25,
        minimum_mean_ratio: float = 0.5,
        maximum_aspect_ratio: float = 4.0,
    ) -> None:
        route_ = parse(route, NativeHexGridRoute, "NativeHexGridSchedule.route")
        scaled = finite_real_scalar(minimum_scaled_jacobian, "minimum_scaled_jacobian")
        mean = finite_real_scalar(minimum_mean_ratio, "minimum_mean_ratio")
        aspect = finite_real_scalar(maximum_aspect_ratio, "maximum_aspect_ratio")
        if not 0.0 < scaled <= 1.0:
            raise ValueError("minimum_scaled_jacobian must lie in (0, 1].")
        if not 0.0 < mean <= 1.0 or aspect < 1.0:
            raise ValueError(
                "Hex quality requires mean ratio in (0, 1] and aspect at least one."
            )
        self.route = route_
        self.minimum_scaled_jacobian = scaled
        self.minimum_mean_ratio = mean
        self.maximum_aspect_ratio = aspect
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "native-hex-grid-schedule",
                "route": route_,
                "minimum_scaled_jacobian": scaled,
                "minimum_mean_ratio": mean,
                "maximum_aspect_ratio": aspect,
            }
        )


__all__ = ["NativeHexGridRoute", "NativeHexGridSchedule"]
