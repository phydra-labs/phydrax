#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._atlas_cover import AtlasCover
from ._density import pullback_density, VolumeDensity
from ._forms import DifferentialForm, pullback_form
from ._map import DifferentiableMap
from ._metric import pullback_metric, RiemannianMetric
from ._tensor import reexpress_tensor, TensorType


class PatchwiseTensorField(StrictModule):
    cover: AtlasCover
    local_fields: tuple[Callable[[Array], Array], ...]
    tensor_type: TensorType

    def __init__(
        self,
        cover: AtlasCover,
        local_fields: Sequence[Callable[[Array], Array]],
        tensor_type: TensorType,
        /,
    ):
        fields = tuple(local_fields)
        if not isinstance(cover, AtlasCover):
            raise TypeError("cover must be an AtlasCover.")
        if len(fields) != len(cover.atlas.charts) or any(
            not callable(field) for field in fields
        ):
            raise ValueError("One callable tensor field is required per chart.")
        if not isinstance(tensor_type, TensorType):
            raise TypeError("tensor_type must be a TensorType.")
        self.cover = cover
        self.local_fields = fields
        self.tensor_type = tensor_type

    def transition_residual(
        self,
        source: int,
        target: int,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        transition = self.cover.atlas.transition(source, target)
        points = jnp.asarray(coordinates)
        source_value = self.local_fields[int(source)](points)
        transformed = reexpress_tensor(
            transition,
            source_value,
            self.tensor_type,
            points,
        )
        target_value = self.local_fields[int(target)](transition(points))
        return jnp.max(jnp.abs(transformed - target_value))


class PatchwiseDifferentialForm(StrictModule):
    cover: AtlasCover
    local_forms: tuple[DifferentialForm, ...]

    def __init__(
        self,
        cover: AtlasCover,
        local_forms: Sequence[DifferentialForm],
        /,
    ):
        forms = tuple(local_forms)
        if not isinstance(cover, AtlasCover):
            raise TypeError("cover must be an AtlasCover.")
        if len(forms) != len(cover.atlas.charts):
            raise ValueError("One differential form is required per chart.")
        degree = forms[0].degree
        for chart, form in zip(cover.atlas.charts, forms, strict=True):
            if not isinstance(form, DifferentialForm):
                raise TypeError("local_forms must contain DifferentialForm objects.")
            if form.degree != degree or not chart.compatible_with(form.chart):
                raise ValueError("Patchwise forms must share degree and chart identity.")
        self.cover = cover
        self.local_forms = forms

    @property
    def degree(self) -> int:
        return self.local_forms[0].degree

    def transition_residual(
        self,
        source: int,
        target: int,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        transition = self.cover.atlas.transition(source, target)
        map = DifferentiableMap(
            transition.source, transition.target, transition.map_function
        )
        source_value = self.local_forms[int(source)](coordinates)
        target_pullback = pullback_form(self.local_forms[int(target)], map)
        return jnp.max(jnp.abs(source_value - target_pullback(coordinates)))


class PatchwiseMetric(StrictModule):
    cover: AtlasCover
    local_metrics: tuple[RiemannianMetric, ...]

    def __init__(
        self,
        cover: AtlasCover,
        local_metrics: Sequence[RiemannianMetric],
        /,
    ):
        metrics = tuple(local_metrics)
        if not isinstance(cover, AtlasCover):
            raise TypeError("cover must be an AtlasCover.")
        if len(metrics) != len(cover.atlas.charts):
            raise ValueError("One metric is required per chart.")
        for chart, metric in zip(cover.atlas.charts, metrics, strict=True):
            if not isinstance(metric, RiemannianMetric) or not chart.compatible_with(
                metric.chart
            ):
                raise ValueError("Patchwise metric chart identities must match.")
        self.cover = cover
        self.local_metrics = metrics

    def transition_residual(
        self,
        source: int,
        target: int,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        transition = self.cover.atlas.transition(source, target)
        pulled = pullback_metric(self.local_metrics[int(target)], transition)
        return jnp.max(
            jnp.abs(self.local_metrics[int(source)](coordinates) - pulled(coordinates))
        )


class PatchwiseDensity(StrictModule):
    cover: AtlasCover
    local_densities: tuple[VolumeDensity, ...]

    def __init__(
        self,
        cover: AtlasCover,
        local_densities: Sequence[VolumeDensity],
        /,
    ):
        densities = tuple(local_densities)
        if not isinstance(cover, AtlasCover):
            raise TypeError("cover must be an AtlasCover.")
        if len(densities) != len(cover.atlas.charts):
            raise ValueError("One density is required per chart.")
        for chart, density in zip(cover.atlas.charts, densities, strict=True):
            if not isinstance(density, VolumeDensity) or not chart.compatible_with(
                density.chart
            ):
                raise ValueError("Patchwise density chart identities must match.")
        self.cover = cover
        self.local_densities = densities

    def transition_residual(
        self,
        source: int,
        target: int,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        transition = self.cover.atlas.transition(source, target)
        pulled = pullback_density(self.local_densities[int(target)], transition)
        return jnp.max(
            jnp.abs(self.local_densities[int(source)](coordinates) - pulled(coordinates))
        )


__all__ = [
    "PatchwiseDensity",
    "PatchwiseDifferentialForm",
    "PatchwiseMetric",
    "PatchwiseTensorField",
]
