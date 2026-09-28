#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Terminal-event policy and regime-transition evidence for radial solves."""

from __future__ import annotations

import equinox as eqx
import numpy as np
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_integer


def _optional_positive(value: float | None, name: str, /) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite or None.")
    return number


class BubbleEventPolicy(StrictModule, NonTrainableState):
    """Terminal events and regime-switching controls of one radial solve.

    - `minimum_radius_ratio`: stop with `MINIMUM_RADIUS` when `R ≤ ratio·R_eq`.
    - `mach_limit`: stop with `MACH_LIMIT` when `|Ṙ|/c ≥ limit` (wall sound
      speed for Gilmore, far-field sound speed otherwise).
    - `hard_core_margin`: stop with `HARD_CORE` when the gas free-volume
      fraction `(V − V_c)/V` falls to this value.
    - `regime_capacity`: maximum number of interface regime transitions.
    - `regime_hysteresis`: dimensionless offset added to every regime guard so
      that a restart exactly on a guard is strictly inside the new regime.
    - `event_tolerance`: root-finding tolerance on nondimensional time.
    - `transversality_tolerance`: minimum nondimensional `|Ṙ|` at a regime
      transition for which derivatives through the event are claimed.

    Invalid thermodynamic/law states (`INVALID_STATE`) and drive support exit
    (`SUPPORT_EXIT`) always terminate.
    """

    minimum_radius_ratio: float | None = eqx.field(static=True)
    mach_limit: float | None = eqx.field(static=True)
    hard_core_margin: float = eqx.field(static=True)
    regime_capacity: int = eqx.field(static=True)
    regime_hysteresis: float = eqx.field(static=True)
    event_tolerance: float = eqx.field(static=True)
    transversality_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_radius_ratio: float | None = None,
        mach_limit: float | None = 1.0,
        hard_core_margin: float = 1.0e-3,
        regime_capacity: int = 64,
        regime_hysteresis: float = 1.0e-7,
        event_tolerance: float = 1.0e-10,
        transversality_tolerance: float = 1.0e-6,
    ) -> None:
        ratio = _optional_positive(minimum_radius_ratio, "minimum_radius_ratio")
        if ratio is not None and ratio >= 1.0:
            raise ValueError("minimum_radius_ratio must be below 1.")
        mach = _optional_positive(mach_limit, "mach_limit")
        margin = float(hard_core_margin)
        if not np.isfinite(margin) or not 0.0 <= margin < 1.0:
            raise ValueError("hard_core_margin must lie in [0, 1).")
        capacity = positive_integer(regime_capacity, "regime_capacity")
        hysteresis = _optional_positive(regime_hysteresis, "regime_hysteresis")
        tolerance = _optional_positive(event_tolerance, "event_tolerance")
        transversality = _optional_positive(transversality_tolerance, "transversality_tolerance")
        if hysteresis is None or tolerance is None or transversality is None:
            raise ValueError("Event tolerances must be positive.")
        self.minimum_radius_ratio = ratio
        self.mach_limit = mach
        self.hard_core_margin = margin
        self.regime_capacity = capacity
        self.regime_hysteresis = hysteresis
        self.event_tolerance = tolerance
        self.transversality_tolerance = transversality
        self.policy_id = canonical_fingerprint(
            {
                "kind": "bubble-event-policy",
                "minimum_radius_ratio": ratio,
                "mach_limit": mach,
                "hard_core_margin": margin,
                "regime_capacity": capacity,
                "regime_hysteresis": hysteresis,
                "event_tolerance": tolerance,
                "transversality_tolerance": transversality,
            }
        )


class BubbleRegimeTape(StrictModule):
    """Fixed-capacity record of localized interface regime transitions.

    Slots beyond `count` are inactive and hold the initial regime and zero time.
    `transversality` is the nondimensional wall speed at each crossing.
    """

    times: Array
    from_regime: Array
    to_regime: Array
    guard: Array
    radius: Array
    transversality: Array
    active: Array
    count: Array
    capacity_exceeded: Array


__all__ = ["BubbleEventPolicy", "BubbleRegimeTape"]
