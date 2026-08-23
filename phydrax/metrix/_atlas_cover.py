#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._atlas import CoordinateAtlas
from ._chart import ChartTransition, CoordinateChart


class ChartSupport(StrictModule):
    """Declared open support predicate for one atlas chart."""

    chart: CoordinateChart
    predicate: Callable[[Array], Array]
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        chart: CoordinateChart,
        predicate: Callable[[Array], Array],
        /,
        *,
        support_id: str,
    ):
        if not isinstance(chart, CoordinateChart):
            raise TypeError("chart must be a CoordinateChart.")
        if not callable(predicate):
            raise TypeError("predicate must be callable.")
        identifier = str(support_id)
        if not identifier:
            raise ValueError("support_id must be non-empty.")
        self.chart = chart
        self.predicate = predicate
        self.support_id = identifier

    def contains(self, coordinates: ArrayLike, /) -> Array:
        points = jnp.asarray(coordinates)
        if points.shape[-1:] != (self.chart.dimension,):
            raise ValueError("Support coordinates must match the chart dimension.")
        result = jnp.asarray(self.predicate(points), dtype=bool)
        if result.shape != points.shape[:-1]:
            raise ValueError("Support predicate must preserve coordinate leading axes.")
        return result


class AtlasOverlap(StrictModule):
    """One directed transition restricted to an explicit overlap support."""

    transition: ChartTransition
    source_support: Callable[[Array], Array]
    overlap_id: str = eqx.field(static=True)
    source_index: int = eqx.field(static=True)
    target_index: int = eqx.field(static=True)

    def __init__(
        self,
        source_index: int,
        target_index: int,
        transition: ChartTransition,
        source_support: Callable[[Array], Array],
        /,
        *,
        overlap_id: str,
    ):
        if not isinstance(transition, ChartTransition):
            raise TypeError("transition must be a ChartTransition.")
        if not callable(source_support):
            raise TypeError("source_support must be callable.")
        identifier = str(overlap_id)
        if not identifier:
            raise ValueError("overlap_id must be non-empty.")
        self.source_index = int(source_index)
        self.target_index = int(target_index)
        self.transition = transition
        self.source_support = source_support
        self.overlap_id = identifier

    def contains(self, coordinates: ArrayLike, /) -> Array:
        points = jnp.asarray(coordinates)
        result = jnp.asarray(self.source_support(points), dtype=bool)
        if result.shape != points.shape[:-1]:
            raise ValueError("Overlap predicate must preserve coordinate leading axes.")
        return result


class AtlasCover(StrictModule):
    """Fixed atlas graph with chart supports and directed overlap domains."""

    atlas: CoordinateAtlas
    supports: tuple[ChartSupport, ...]
    overlaps: tuple[AtlasOverlap, ...]
    cover_id: str = eqx.field(static=True)

    def __init__(
        self,
        atlas: CoordinateAtlas,
        supports: Sequence[ChartSupport],
        overlaps: Sequence[AtlasOverlap],
        /,
        *,
        cover_id: str,
    ):
        if not isinstance(atlas, CoordinateAtlas):
            raise TypeError("atlas must be a CoordinateAtlas.")
        supports_ = tuple(supports)
        if len(supports_) != len(atlas.charts):
            raise ValueError("One support is required for every atlas chart.")
        for chart, support in zip(atlas.charts, supports_, strict=True):
            if not isinstance(support, ChartSupport) or not chart.compatible_with(
                support.chart
            ):
                raise ValueError("Atlas chart and support identities must match.")
        overlaps_ = tuple(overlaps)
        seen = set()
        for overlap in overlaps_:
            if not isinstance(overlap, AtlasOverlap):
                raise TypeError("overlaps must contain AtlasOverlap objects.")
            pair = (overlap.source_index, overlap.target_index)
            if pair in seen:
                raise ValueError("Atlas overlaps must be unique by direction.")
            seen.add(pair)
            transition = atlas.transition(*pair)
            if not transition.source.compatible_with(
                overlap.transition.source
            ) or not transition.target.compatible_with(overlap.transition.target):
                raise ValueError("Overlap transition does not match the atlas graph.")
        identifier = str(cover_id)
        if not identifier:
            raise ValueError("cover_id must be non-empty.")
        self.atlas = atlas
        self.supports = supports_
        self.overlaps = overlaps_
        self.cover_id = identifier

    def support(self, chart_index: int, coordinates: ArrayLike, /) -> Array:
        return self.supports[int(chart_index)].contains(coordinates)

    def overlap(
        self,
        source_index: int,
        target_index: int,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        pair = (int(source_index), int(target_index))
        for overlap in self.overlaps:
            if (overlap.source_index, overlap.target_index) == pair:
                return overlap.contains(coordinates)
        raise ValueError("Atlas cover has no declared directed overlap.")

    def cocycle_residual(
        self,
        first: int,
        second: int,
        third: int,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        direct = self.atlas.transition(first, third)
        composed = self.atlas.transition(first, second).compose(
            self.atlas.transition(second, third)
        )
        points = jnp.asarray(coordinates)
        return jnp.max(jnp.abs(direct(points) - composed(points)))


__all__ = ["AtlasCover", "AtlasOverlap", "ChartSupport"]
