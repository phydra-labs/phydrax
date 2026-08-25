#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._chart import CoordinateChart
from .._exterior_basis import axes_bitmap
from .._forms import DifferentialForm
from .._metric import (
    AbstractSemiRiemannianMetric,
    MetricSignature,
    RiemannianMetric,
    SemiRiemannianMetric,
)
from ._blades import CliffordBladeLayout
from ._spec import CliffordAlgebraSpec


class _ConstantCliffordMetricMap(StrictModule, NonTrainableState):
    diagonal: Array

    def __init__(self, diagonal: tuple[int, ...], /):
        self.diagonal = jnp.asarray(diagonal, dtype=float)

    def __call__(self, coordinates: Array, /) -> Array:
        return jnp.diag(self.diagonal).astype(coordinates.dtype)


class _EmbeddedFormField(StrictModule):
    form: DifferentialForm
    layout: CliffordBladeLayout
    output_positions: Array
    index_scales: Array

    def __init__(
        self,
        form: DifferentialForm,
        layout: CliffordBladeLayout,
        output_positions: tuple[int, ...],
        index_scales: tuple[int, ...],
        /,
    ):
        self.form = form
        self.layout = layout
        self.output_positions = jnp.asarray(output_positions, dtype=jnp.int32)
        self.index_scales = jnp.asarray(index_scales, dtype=jnp.int8)

    def __call__(self, coordinates: Array, /) -> Array:
        coefficients = self.form._coefficients_point(coordinates)
        values = coefficients * self.index_scales.astype(coefficients.dtype)
        return (
            jnp.zeros((self.layout.blade_count,), dtype=values.dtype)
            .at[self.output_positions]
            .set(values)
        )


class _ExtractedFormCoefficient(StrictModule):
    field: Callable[[Array], Array]
    layout: CliffordBladeLayout
    source_positions: Array
    index_scales: Array

    def __init__(
        self,
        field: Callable[[Array], Array],
        layout: CliffordBladeLayout,
        source_positions: tuple[int, ...],
        index_scales: tuple[int, ...],
        /,
    ):
        self.field = field
        self.layout = layout
        self.source_positions = jnp.asarray(source_positions, dtype=jnp.int32)
        self.index_scales = jnp.asarray(index_scales, dtype=jnp.int8)

    def __call__(self, coordinates: Array, /) -> Array:
        values = jnp.asarray(self.field(coordinates))
        if values.shape != (self.layout.blade_count,):
            raise ValueError(
                "Pointwise Clifford field must return its declared blade count."
            )
        selected = values[self.source_positions]
        return selected * self.index_scales.astype(selected.dtype)


class CliffordMetricBridge(StrictModule, NonTrainableState):
    """Exact bridge between a constant orthogonal frame and differential forms."""

    algebra: CliffordAlgebraSpec
    chart: CoordinateChart
    layout: CliffordBladeLayout
    metric: AbstractSemiRiemannianMetric
    bridge_id: str = eqx.field(static=True)

    def __init__(
        self,
        algebra: CliffordAlgebraSpec,
        chart: CoordinateChart,
        /,
    ):
        if not isinstance(algebra, CliffordAlgebraSpec):
            raise TypeError("algebra must be a CliffordAlgebraSpec.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("chart must be a CoordinateChart.")
        if chart.dimension != algebra.dimension:
            raise ValueError("Clifford algebra and form chart dimensions do not match.")
        if not algebra.nondegenerate:
            raise ValueError(
                "Differential-form Clifford bridge requires a nondegenerate metric."
            )
        metric_map = _ConstantCliffordMetricMap(algebra.diagonal)
        if algebra.positive_definite:
            metric: AbstractSemiRiemannianMetric = RiemannianMetric(
                metric_map,
                chart=chart,
            )
        else:
            metric = SemiRiemannianMetric(
                metric_map,
                chart=chart,
                signature=MetricSignature(algebra.positive, algebra.negative),
            )
        layout = CliffordBladeLayout.full(algebra)
        self.algebra = algebra
        self.chart = chart
        self.layout = layout
        self.metric = metric
        self.bridge_id = canonical_fingerprint(
            {
                "kind": "clifford-form-bridge-v1",
                "algebra": algebra.algebra_id,
                "orientation": algebra.orientation,
                "chart": chart.name,
                "coordinates": list(chart.coordinates),
                "layout": layout.layout_id,
            }
        )

    def _grade_map(
        self, form: DifferentialForm, /
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not isinstance(form, DifferentialForm):
            raise TypeError("Expected a DifferentialForm.")
        if not form.chart.compatible_with(self.chart):
            raise ValueError("Differential form uses an incompatible coordinate chart.")
        positions = []
        scales = []
        for axes in form.indices:
            bitmap = axes_bitmap(axes, self.algebra.dimension)
            positions.append(self.layout.position(bitmap))
            scale = 1
            for axis in axes:
                scale *= self.algebra.diagonal[axis]
            scales.append(scale)
        return tuple(positions), tuple(scales)

    def embed(self, form: DifferentialForm, /) -> Callable[[Array], Array]:
        """Raise a homogeneous form and embed it into the full blade layout."""
        positions, scales = self._grade_map(form)
        return _EmbeddedFormField(form, self.layout, positions, scales)

    def extract(
        self,
        field: Callable[[Array], Array],
        degree: int,
        /,
    ) -> DifferentialForm:
        """Lower one Clifford grade into a homogeneous differential form."""
        if not callable(field):
            raise TypeError("field must be callable.")
        degree_ = int(degree)
        template = DifferentialForm(
            lambda coordinates: jnp.zeros(
                (len(tuple(self.layout.grade_positions(degree_))),),
                dtype=coordinates.dtype,
            ),
            chart=self.chart,
            degree=degree_,
        )
        positions, scales = self._grade_map(template)
        return DifferentialForm(
            _ExtractedFormCoefficient(field, self.layout, positions, scales),
            chart=self.chart,
            degree=degree_,
        )


__all__ = ["CliffordMetricBridge"]
