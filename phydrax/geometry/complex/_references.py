#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from ..._strict import StrictModule
from ...metrix import (
    ComplexCoordinateConvention,
    CoordinateChart,
    euclidean_metric,
    HermitianStructure,
    KahlerStructure,
    RiemannianMetric,
    standard_complex_structure,
)


class FlatComplexTorus(StrictModule):
    """Flat local geometry and explicit periods for a complex torus."""

    convention: ComplexCoordinateConvention
    metric: RiemannianMetric
    kahler: KahlerStructure
    complex_dimension: int = eqx.field(static=True)
    period: float = eqx.field(static=True)

    def __init__(self, complex_dimension: int, /, *, period: float = 1.0):
        dimension = int(complex_dimension)
        if dimension < 1 or float(period) <= 0.0:
            raise ValueError("Complex dimension and period must be positive.")
        chart = CoordinateChart(
            f"complex-torus:{dimension}",
            tuple(
                [f"x{axis}" for axis in range(dimension)]
                + [f"y{axis}" for axis in range(dimension)]
            ),
        )
        convention = ComplexCoordinateConvention(chart)
        metric = euclidean_metric(chart)
        self.convention = convention
        self.metric = metric
        self.kahler = KahlerStructure(
            HermitianStructure(metric, standard_complex_structure(convention))
        )
        self.complex_dimension = dimension
        self.period = float(period)


__all__ = ["FlatComplexTorus"]
