#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable numerical schedules independent of native execution routes."""

from __future__ import annotations

from typing import final

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import nonnegative_integer, positive_integer
from ...typing import checked


@final
class NativeCurveSchedule(StrictModule, NonTrainableState):
    """Numerical schedule of arc-length curve discretization.

    ``quadrature_intervals`` and ``quadrature_order`` set the composite
    Gauss-Legendre rule of arc-length and size-density integration per curve;
    ``fidelity_samples`` is the number of interior samples of each interval at
    which chord deviation is measured.
    """

    quadrature_intervals: int = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    fidelity_samples: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        quadrature_intervals: int = 64,
        quadrature_order: int = 8,
        fidelity_samples: int = 8,
    ) -> None:
        intervals = positive_integer(
            quadrature_intervals, "NativeCurveSchedule.quadrature_intervals"
        )
        order = positive_integer(quadrature_order, "NativeCurveSchedule.quadrature_order")
        samples = positive_integer(
            fidelity_samples, "NativeCurveSchedule.fidelity_samples"
        )
        self.quadrature_intervals = intervals
        self.quadrature_order = order
        self.fidelity_samples = samples
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "native-curve-schedule",
                "quadrature": [intervals, order],
                "fidelity_samples": samples,
            }
        )


@final
class NativeSurfaceSchedule(StrictModule, NonTrainableState):
    """Numerical schedule of chart-space surface generation.

    ``quality_angle_degrees`` is the physical minimum-angle aim of refinement
    (a requested hard quality target is enforced by compliance, never relaxed
    by the aim); ``maximum_rounds`` bounds the refinement batches;
    ``spacing_fraction`` of the local size is the smallest physical distance
    at which a refinement point may approach an existing vertex;
    ``curve_schedule`` discretizes the shared feature curves.
    """

    quality_angle_degrees: float = eqx.field(static=True)
    maximum_rounds: int = eqx.field(static=True)
    spacing_fraction: float = eqx.field(static=True)
    curve_schedule: NativeCurveSchedule
    schedule_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        quality_angle_degrees: float = 22.0,
        maximum_rounds: int = 64,
        spacing_fraction: float = 0.25,
        curve_schedule: NativeCurveSchedule | None = None,
    ) -> None:
        angle = float(quality_angle_degrees)
        spacing = float(spacing_fraction)
        if not np.isfinite(angle) or not 0.0 <= angle < 34.0:
            raise ValueError(
                "NativeSurfaceSchedule.quality_angle_degrees must lie in [0, 34); "
                f"got {angle!r}"
            )
        if not np.isfinite(spacing) or not 0.0 < spacing < 1.0:
            raise ValueError(
                "NativeSurfaceSchedule.spacing_fraction must lie in (0, 1); "
                f"got {spacing!r}"
            )
        rounds = positive_integer(maximum_rounds, "NativeSurfaceSchedule.maximum_rounds")
        curves = NativeCurveSchedule() if curve_schedule is None else curve_schedule
        self.quality_angle_degrees = angle
        self.maximum_rounds = rounds
        self.spacing_fraction = spacing
        self.curve_schedule = curves
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "native-surface-schedule",
                "quality_angle_degrees": angle,
                "maximum_rounds": rounds,
                "spacing_fraction": spacing,
                "curve_schedule": curves.schedule_id,
            }
        )


@final
class NativeStructuredSchedule(StrictModule, NonTrainableState):
    """Bounded variational optimization of declared transfinite blocks."""

    optimization_steps: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, *, optimization_steps: int = 0) -> None:
        steps = nonnegative_integer(
            optimization_steps, "NativeStructuredSchedule.optimization_steps"
        )
        self.optimization_steps = steps
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "native-structured-schedule",
                "optimization_steps": steps,
            }
        )


__all__ = ["NativeCurveSchedule", "NativeSurfaceSchedule", "NativeStructuredSchedule"]
