#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._polynomial._cubature import CubatureReference
from .._strict import StrictModule


CubatureComponent: TypeAlias = Literal["interior", "boundary"]


class CubatureMapEvaluation(StrictModule):
    """One batched physical map evaluation with measure and admissibility evidence."""

    points: Array
    measure_scale: Array
    admissible: Array
    orientation: Array
    normal: Array | None

    def __init__(
        self,
        points: Array,
        measure_scale: Array,
        admissible: Array,
        orientation: Array,
        normal: Array | None = None,
        /,
    ):
        points_ = jnp.asarray(points)
        scale = jnp.asarray(measure_scale, dtype=jnp.real(points_).dtype)
        admissible_ = jnp.asarray(admissible, dtype=bool)
        orientation_ = jnp.asarray(orientation, dtype=scale.dtype)
        if points_.shape[:-1] != scale.shape:
            raise ValueError("measure_scale must match mapped-point leading dimensions.")
        if admissible_.shape != scale.shape or orientation_.shape != scale.shape:
            raise ValueError("admissible and orientation must match measure_scale.")
        normal_ = None if normal is None else jnp.asarray(normal, dtype=points_.dtype)
        if normal_ is not None and normal_.shape != points_.shape:
            raise ValueError("normal must match mapped points when supplied.")
        self.points = points_
        self.measure_scale = scale
        self.admissible = admissible_
        self.orientation = orientation_
        self.normal = normal_


_REFERENCE_DIMENSION = {
    "triangle": 2,
    "tetrahedron": 3,
    "circle": 2,
    "disk": 2,
    "sphere": 3,
    "ball": 3,
}


class AbstractCubatureMap(StrictModule):
    """Abstract batched map from one canonical cubature reference."""

    @property
    @abstractmethod
    def num_charts(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def reference_domain(self) -> CubatureReference:
        raise NotImplementedError

    @property
    def reference_dimension(self) -> int:
        return _REFERENCE_DIMENSION[self.reference_domain]

    @property
    @abstractmethod
    def ambient_dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def reference_mask(self, chart_indices: Array, reference: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        chart_indices: Array,
        reference: Array,
        /,
    ) -> CubatureMapEvaluation:
        """Evaluate the legacy map contract as one fail-closed geometry product."""

        points = self.map(chart_indices, reference)
        measure_scale = jnp.asarray(
            self.jacobian(chart_indices, reference),
            dtype=jnp.real(points).dtype,
        )
        if measure_scale.shape != points.shape[:-1]:
            raise ValueError(
                "Legacy CubatureMap.jacobian must return scalar measure scale "
                "matching mapped-point leading dimensions."
            )
        orientation = jnp.ones_like(measure_scale)
        mask = self.reference_mask(chart_indices, reference)
        admissible = (
            jnp.asarray(mask, dtype=bool)
            & jnp.all(jnp.isfinite(points), axis=-1)
            & jnp.isfinite(measure_scale)
            & (measure_scale > 0)
            & (orientation != 0)
        )
        return CubatureMapEvaluation(
            points,
            measure_scale,
            admissible,
            orientation,
        )


class CubatureAtlas(StrictModule):
    """Collection of physical charts sharing one canonical cubature reference."""

    mapping: AbstractCubatureMap
    source_entity_ids: Array
    source_id: str = eqx.field(static=True)
    physical_tags: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        mapping: AbstractCubatureMap,
        *,
        source_entity_ids: Array,
        source_id: str,
        physical_tags: Sequence[str] | None = None,
    ):
        if not isinstance(mapping, AbstractCubatureMap):
            raise TypeError("CubatureAtlas mapping must be an AbstractCubatureMap.")
        entity_ids = jnp.asarray(source_entity_ids, dtype=jnp.int32).reshape((-1,))
        if entity_ids.shape != (mapping.num_charts,):
            raise ValueError("source_entity_ids must contain one ID per cubature chart.")
        tags = (
            tuple("cubature" for _ in range(mapping.num_charts))
            if physical_tags is None
            else tuple(physical_tags)
        )
        if len(tags) != mapping.num_charts or any(not tag for tag in tags):
            raise ValueError("physical_tags must contain one nonempty tag per chart.")
        if not source_id:
            raise ValueError("CubatureAtlas source_id must be nonempty.")
        self.mapping = mapping
        self.source_entity_ids = entity_ids
        self.source_id = source_id
        self.physical_tags = tags

    @property
    def num_charts(self) -> int:
        return self.mapping.num_charts

    @property
    def reference_domain(self) -> CubatureReference:
        return self.mapping.reference_domain

    @property
    def reference_dimension(self) -> int:
        return self.mapping.reference_dimension

    @property
    def ambient_dimension(self) -> int:
        return self.mapping.ambient_dimension

    def _validate_inputs(self, chart_indices: Array, reference: Array):
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        reference_ = jnp.asarray(reference, dtype=float)
        if reference_.shape[:-1] != indices.shape:
            raise ValueError("chart_indices must match reference leading dimensions.")
        if reference_.shape[-1] != self.reference_dimension:
            raise ValueError(
                f"reference must have trailing dimension {self.reference_dimension}."
            )
        return indices, reference_

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        indices, reference_ = self._validate_inputs(chart_indices, reference)
        return self.mapping.map(indices, reference_)

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        indices, reference_ = self._validate_inputs(chart_indices, reference)
        return self.mapping.jacobian(indices, reference_)

    def reference_mask(self, chart_indices: Array, reference: Array, /) -> Array:
        indices, reference_ = self._validate_inputs(chart_indices, reference)
        return self.mapping.reference_mask(indices, reference_)

    def evaluate(
        self,
        chart_indices: Array,
        reference: Array,
        /,
    ) -> CubatureMapEvaluation:
        indices, reference_ = self._validate_inputs(chart_indices, reference)
        return self.mapping.evaluate(indices, reference_)

    def select(
        self,
        *,
        entity_ids: Sequence[int] | None = None,
        tags: Sequence[str] | None = None,
    ) -> CubatureAtlas:
        mask = np.ones((self.num_charts,), dtype=bool)
        if entity_ids is not None:
            mask &= np.isin(
                np.asarray(self.source_entity_ids),
                np.asarray(tuple(entity_ids), dtype=np.int32),
            )
        if tags is not None:
            selected_tags = frozenset(tags)
            mask &= np.asarray([tag in selected_tags for tag in self.physical_tags])
        selected = np.flatnonzero(mask).astype(np.int32)
        if selected.size == 0:
            raise ValueError("CubatureAtlas selection contains no charts.")
        return CubatureAtlas(
            _SelectedCubatureMap(self.mapping, jnp.asarray(selected)),
            source_entity_ids=self.source_entity_ids[selected],
            source_id=self.source_id,
            physical_tags=tuple(self.physical_tags[index] for index in selected),
        )

    def translated(self, offset: Array, /) -> CubatureAtlas:
        return CubatureAtlas(
            _TranslatedCubatureMap(self.mapping, offset),
            source_entity_ids=self.source_entity_ids,
            source_id=self.source_id,
            physical_tags=self.physical_tags,
        )


class _SelectedCubatureMap(AbstractCubatureMap):
    base: AbstractCubatureMap
    chart_indices: Array

    def __init__(self, base: AbstractCubatureMap, chart_indices: Array):
        self.base = base
        self.chart_indices = jnp.asarray(chart_indices, dtype=jnp.int32).reshape((-1,))

    @property
    def num_charts(self) -> int:
        return int(self.chart_indices.shape[0])

    @property
    def reference_domain(self) -> CubatureReference:
        return self.base.reference_domain

    @property
    def ambient_dimension(self) -> int:
        return self.base.ambient_dimension

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.map(self.chart_indices[chart_indices], reference)

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.jacobian(self.chart_indices[chart_indices], reference)

    def reference_mask(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.reference_mask(self.chart_indices[chart_indices], reference)

    def evaluate(
        self,
        chart_indices: Array,
        reference: Array,
        /,
    ) -> CubatureMapEvaluation:
        return self.base.evaluate(self.chart_indices[chart_indices], reference)


class _TranslatedCubatureMap(AbstractCubatureMap):
    base: AbstractCubatureMap
    offset: Array

    def __init__(self, base: AbstractCubatureMap, offset: Array):
        self.base = base
        self.offset = jnp.asarray(offset, dtype=float)

    @property
    def num_charts(self) -> int:
        return self.base.num_charts

    @property
    def reference_domain(self) -> CubatureReference:
        return self.base.reference_domain

    @property
    def ambient_dimension(self) -> int:
        return self.base.ambient_dimension

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.map(chart_indices, reference) + self.offset

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.jacobian(chart_indices, reference)

    def reference_mask(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.reference_mask(chart_indices, reference)

    def evaluate(
        self,
        chart_indices: Array,
        reference: Array,
        /,
    ) -> CubatureMapEvaluation:
        evaluation = self.base.evaluate(chart_indices, reference)
        return CubatureMapEvaluation(
            evaluation.points + self.offset,
            evaluation.measure_scale,
            evaluation.admissible,
            evaluation.orientation,
            evaluation.normal,
        )


@runtime_checkable
class CubatureAtlasProvider(Protocol):
    """Structural provider of physical cubature charts."""

    def cubature_atlas(
        self, component: Literal["interior", "boundary"], /
    ) -> CubatureAtlas: ...


__all__ = [
    "AbstractCubatureMap",
    "CubatureMapEvaluation",
    "CubatureAtlas",
    "CubatureAtlasProvider",
    "CubatureComponent",
]
