#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable numerical schedules for native curve, surface, and block routes."""

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
    """Quadrature and diagnostic sampling schedule for curve placement."""

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
                "quadrature": (intervals, order),
                "fidelity_samples": samples,
            }
        )


@final
class NativeSurfaceSchedule(StrictModule, NonTrainableState):
    """Physical quality aim and spacing used to schedule surface refinement.

    The schedule contains no refinement-round ceiling: unresolved physical
    violations terminate through numerical stagnation/representability status.
    ``normal_deviation_aim`` (radians) also refines cells whose facet turns from
    sampled source normals by more than the aim; it steers refinement only and
    is not a certified bound.
    """

    quality_angle_degrees: float = eqx.field(static=True)
    spacing_fraction: float = eqx.field(static=True)
    minimum_edge_samples: int = eqx.field(static=True)
    normal_deviation_aim: float | None = eqx.field(static=True)
    curve_schedule: NativeCurveSchedule
    schedule_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        quality_angle_degrees: float = 22.0,
        spacing_fraction: float = 0.9,
        minimum_edge_samples: int = 2,
        normal_deviation_aim: float | None = None,
        curve_schedule: NativeCurveSchedule | None = None,
    ) -> None:
        angle = float(quality_angle_degrees)
        spacing = float(spacing_fraction)
        if not np.isfinite(angle) or not 0.0 <= angle < 34.0:
            raise ValueError(
                "NativeSurfaceSchedule.quality_angle_degrees must lie in [0, 34)."
            )
        if not np.isfinite(spacing) or not 0.0 < spacing < 1.0:
            raise ValueError("NativeSurfaceSchedule.spacing_fraction must lie in (0, 1).")
        samples = positive_integer(
            minimum_edge_samples, "NativeSurfaceSchedule.minimum_edge_samples"
        )
        if samples < 2:
            raise ValueError(
                "minimum_edge_samples counts distinct interval endpoints and must be at least two."
            )
        aim = None if normal_deviation_aim is None else float(normal_deviation_aim)
        if aim is not None and (not np.isfinite(aim) or not 0.0 < aim <= np.pi):
            raise ValueError(
                "NativeSurfaceSchedule.normal_deviation_aim must lie in (0, pi]."
            )
        curves = NativeCurveSchedule() if curve_schedule is None else curve_schedule
        if not isinstance(curves, NativeCurveSchedule):
            raise TypeError("curve_schedule must be a NativeCurveSchedule.")
        self.quality_angle_degrees = angle
        self.spacing_fraction = spacing
        self.minimum_edge_samples = samples
        self.normal_deviation_aim = aim
        self.curve_schedule = curves
        identity: dict[str, object] = {
            "kind": "native-surface-schedule",
            "quality_angle_degrees": angle,
            "spacing_fraction": spacing,
            "minimum_edge_samples": samples,
            "curve_schedule": curves.schedule_id,
        }
        if aim is not None:
            identity["normal_deviation_aim"] = aim
        self.schedule_id = canonical_fingerprint(identity)


@final
class NativeStructuredSchedule(StrictModule, NonTrainableState):
    """Deterministic variational iteration schedule for transfinite blocks."""

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
