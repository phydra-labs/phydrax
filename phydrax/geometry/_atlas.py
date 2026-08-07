#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._strict import StrictModule


class AbstractBoundaryMap(StrictModule):
    """Abstract batched map from uniform reference cells to a boundary."""

    @property
    @abstractmethod
    def num_charts(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def reference_dimension(self) -> int:
        raise NotImplementedError

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


class TrimDomain(StrictModule):
    """Oriented polygonal trim domain in a two-dimensional chart."""

    outer: Array
    holes: tuple[Array, ...]

    def __init__(self, outer: Array, holes: Sequence[Array] = ()):
        outer_host = np.asarray(outer, dtype=float)
        hole_hosts = tuple(np.asarray(hole, dtype=float) for hole in holes)
        if outer_host.ndim != 2 or outer_host.shape[1] != 2 or outer_host.shape[0] < 3:
            raise ValueError("TrimDomain.outer must have shape (num_points >= 3, 2).")
        if any(
            hole.ndim != 2 or hole.shape[1] != 2 or hole.shape[0] < 3
            for hole in hole_hosts
        ):
            raise ValueError("Every trim hole must have shape (num_points >= 3, 2).")
        self.outer = jnp.asarray(outer_host, dtype=float)
        self.holes = tuple(jnp.asarray(hole, dtype=float) for hole in hole_hosts)

    @staticmethod
    def _inside_loop(points: Array, loop: Array) -> Array:
        start = loop
        end = jnp.roll(loop, -1, axis=0)
        x = points[..., 0, None]
        y = points[..., 1, None]
        crossing = (start[:, 1] > y) != (end[:, 1] > y)
        intersection = (end[:, 0] - start[:, 0]) * (y - start[:, 1]) / jnp.where(
            end[:, 1] != start[:, 1],
            end[:, 1] - start[:, 1],
            1.0,
        ) + start[:, 0]
        return jnp.sum(crossing & (x < intersection), axis=-1) % 2 == 1

    def contains(self, reference: Array, /) -> Array:
        reference_ = jnp.asarray(reference, dtype=float)
        inside = self._inside_loop(reference_, self.outer)
        for hole in self.holes:
            inside &= ~self._inside_loop(reference_, hole)
        return inside


class BoundaryFrame(StrictModule):
    """Physical chart frame evaluated at one or more reference points."""

    origin: Array
    tangents: Array
    normal: Array
    jacobian: Array

    def __init__(self, *, origin: Array, tangents: Array, normal: Array, jacobian: Array):
        self.origin = jnp.asarray(origin, dtype=float)
        self.tangents = jnp.asarray(tangents, dtype=float)
        self.normal = jnp.asarray(normal, dtype=float)
        self.jacobian = jnp.asarray(jacobian, dtype=float)


class BoundaryAtlas(StrictModule):
    """Representation-independent collection of oriented boundary charts."""

    mapping: AbstractBoundaryMap
    source_entity_ids: Array
    source_id: str = eqx.field(static=True)
    physical_tags: tuple[str, ...] = eqx.field(static=True)
    orientation: Array
    seam_owner: Array
    trim_domains: tuple[TrimDomain | None, ...]

    def __init__(
        self,
        mapping: AbstractBoundaryMap,
        *,
        source_entity_ids: Array,
        source_id: str,
        physical_tags: Sequence[str] | None = None,
        orientation: Array | None = None,
        seam_owner: Array | None = None,
        trim_domains: Sequence[TrimDomain | None] | None = None,
    ):
        entity_ids = jnp.asarray(source_entity_ids, dtype=jnp.int32).reshape((-1,))
        if entity_ids.shape != (mapping.num_charts,):
            raise ValueError("source_entity_ids must contain one ID per boundary chart.")
        tags = (
            tuple("boundary" for _ in range(mapping.num_charts))
            if physical_tags is None
            else tuple(physical_tags)
        )
        if len(tags) != mapping.num_charts or any(not tag for tag in tags):
            raise ValueError("physical_tags must contain one non-empty tag per chart.")
        orientation_ = (
            jnp.ones((mapping.num_charts,), dtype=float)
            if orientation is None
            else jnp.asarray(orientation, dtype=float).reshape((-1,))
        )
        if orientation_.shape != (mapping.num_charts,):
            raise ValueError("orientation must contain one sign per chart.")
        orientation_host = np.asarray(orientation_)
        if np.any((orientation_host != 1.0) & (orientation_host != -1.0)):
            raise ValueError("orientation entries must be +1 or -1.")
        seam_owner_ = (
            jnp.ones((mapping.num_charts,), dtype=bool)
            if seam_owner is None
            else jnp.asarray(seam_owner, dtype=bool).reshape((-1,))
        )
        if seam_owner_.shape != (mapping.num_charts,):
            raise ValueError("seam_owner must contain one flag per chart.")
        trims = (
            tuple(None for _ in range(mapping.num_charts))
            if trim_domains is None
            else tuple(trim_domains)
        )
        if len(trims) != mapping.num_charts or any(
            trim is not None and not isinstance(trim, TrimDomain) for trim in trims
        ):
            raise ValueError(
                "trim_domains must contain one TrimDomain or None per chart."
            )
        if not source_id:
            raise ValueError("BoundaryAtlas.source_id must be non-empty.")
        self.mapping = mapping
        self.source_entity_ids = entity_ids
        self.source_id = source_id
        self.physical_tags = tags
        self.orientation = orientation_
        self.seam_owner = seam_owner_
        self.trim_domains = trims

    @property
    def num_charts(self) -> int:
        return self.mapping.num_charts

    @property
    def reference_dimension(self) -> int:
        return self.mapping.reference_dimension

    @property
    def reference_dim(self) -> int:
        """Compatibility spelling used by integration reference rules."""
        return self.reference_dimension

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
        """Return whether reference points lie inside their charts' trim domains."""
        indices, reference_ = self._validate_inputs(chart_indices, reference)
        if self.reference_dimension != 2 or all(
            trim is None for trim in self.trim_domains
        ):
            return jnp.ones(indices.shape, dtype=bool)
        flat_indices = indices.reshape((-1,))
        flat_reference = reference_.reshape((-1, self.reference_dimension))
        branches = tuple(
            (
                (lambda coordinate: jnp.asarray(True))
                if trim is None
                else (lambda coordinate, trim=trim: trim.contains(coordinate))
            )
            for trim in self.trim_domains
        )
        values = jax.vmap(
            lambda index, coordinate: jax.lax.switch(index, branches, coordinate)
        )(flat_indices, flat_reference)
        return values.reshape(indices.shape)

    def differential(self, chart_indices: Array, reference: Array, /) -> Array:
        """Return chart derivatives with shape ``(..., ambient_dim, reference_dim)``."""
        indices, reference_ = self._validate_inputs(chart_indices, reference)
        leading = indices.shape
        flat_indices = indices.reshape((-1,))
        flat_reference = reference_.reshape((-1, self.reference_dimension))
        differential = jax.vmap(
            lambda index, coordinate: jax.jacfwd(
                lambda value: self.mapping.map(index, value)
            )(coordinate)
        )(flat_indices, flat_reference)
        return differential.reshape(
            (*leading, self.ambient_dimension, self.reference_dimension)
        )

    def frame(self, chart_indices: Array, reference: Array, /) -> BoundaryFrame:
        indices, reference_ = self._validate_inputs(chart_indices, reference)
        origin = self.mapping.map(indices, reference_)
        differential = self.differential(indices, reference_)
        if self.reference_dimension == 1 and self.ambient_dimension == 2:
            tangent = differential[..., :, 0]
            tangent = tangent / jnp.linalg.norm(tangent, axis=-1, keepdims=True)
            normal = jnp.stack((tangent[..., 1], -tangent[..., 0]), axis=-1)
            tangents = tangent[..., None, :]
        elif self.reference_dimension == 2 and self.ambient_dimension == 3:
            first = differential[..., :, 0]
            second = differential[..., :, 1]
            first_unit = first / jnp.linalg.norm(first, axis=-1, keepdims=True)
            second_orthogonal = (
                second - jnp.sum(second * first_unit, axis=-1, keepdims=True) * first_unit
            )
            second_unit = second_orthogonal / jnp.linalg.norm(
                second_orthogonal, axis=-1, keepdims=True
            )
            tangents = jnp.stack((first_unit, second_unit), axis=-2)
            normal = jnp.cross(first_unit, second_unit)
        else:
            raise NotImplementedError(
                "Boundary frames currently support curves in 2D and surfaces in 3D."
            )
        normal = normal * self.orientation[indices][..., None]
        return BoundaryFrame(
            origin=origin,
            tangents=tangents,
            normal=normal,
            jacobian=self.mapping.jacobian(indices, reference_),
        )

    def select(
        self,
        *,
        entity_ids: Sequence[int] | None = None,
        tags: Sequence[str] | None = None,
    ) -> BoundaryAtlas:
        """Select charts by source entity ID and/or physical tag."""
        mask = np.ones((self.num_charts,), dtype=bool)
        if entity_ids is not None:
            mask &= np.isin(
                np.asarray(self.source_entity_ids),
                np.asarray(tuple(entity_ids), dtype=np.int32),
            )
        if tags is not None:
            selected_tags = frozenset(tags)
            mask &= np.asarray([tag in selected_tags for tag in self.physical_tags])
        chart_indices = np.flatnonzero(mask).astype(np.int32)
        if chart_indices.size == 0:
            raise ValueError("BoundaryAtlas selection contains no charts.")
        return BoundaryAtlas(
            _SelectedBoundaryMap(self.mapping, jnp.asarray(chart_indices)),
            source_entity_ids=self.source_entity_ids[chart_indices],
            source_id=self.source_id,
            physical_tags=tuple(self.physical_tags[index] for index in chart_indices),
            orientation=self.orientation[chart_indices],
            seam_owner=self.seam_owner[chart_indices],
            trim_domains=tuple(self.trim_domains[index] for index in chart_indices),
        )

    def translated(self, offset: Array, /) -> BoundaryAtlas:
        return BoundaryAtlas(
            _TranslatedBoundaryMap(self.mapping, offset),
            source_entity_ids=self.source_entity_ids,
            source_id=self.source_id,
            physical_tags=self.physical_tags,
            orientation=self.orientation,
            seam_owner=self.seam_owner,
            trim_domains=self.trim_domains,
        )


class _SelectedBoundaryMap(AbstractBoundaryMap):
    base: AbstractBoundaryMap
    chart_indices: Array

    def __init__(self, base: AbstractBoundaryMap, chart_indices: Array):
        self.base = base
        self.chart_indices = jnp.asarray(chart_indices, dtype=jnp.int32).reshape((-1,))

    @property
    def num_charts(self) -> int:
        return self.chart_indices.shape[0]

    @property
    def reference_dimension(self) -> int:
        return self.base.reference_dimension

    @property
    def ambient_dimension(self) -> int:
        return self.base.ambient_dimension

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.map(self.chart_indices[chart_indices], reference)

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.jacobian(self.chart_indices[chart_indices], reference)


class _TranslatedBoundaryMap(AbstractBoundaryMap):
    base: AbstractBoundaryMap
    offset: Array

    def __init__(self, base: AbstractBoundaryMap, offset: Array):
        offset_ = jnp.asarray(offset, dtype=float).reshape((-1,))
        if offset_.shape != (base.ambient_dimension,):
            raise ValueError(
                f"Translation offset must have shape ({base.ambient_dimension},)."
            )
        self.base = base
        self.offset = offset_

    @property
    def num_charts(self) -> int:
        return self.base.num_charts

    @property
    def reference_dimension(self) -> int:
        return self.base.reference_dimension

    @property
    def ambient_dimension(self) -> int:
        return self.base.ambient_dimension

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.map(chart_indices, reference) + self.offset

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.jacobian(chart_indices, reference)


class _CircleBoundaryMap(AbstractBoundaryMap):
    center: Array
    radius: Array

    def __init__(self, center: Array, radius: Array):
        self.center = jnp.asarray(center, dtype=float).reshape((2,))
        self.radius = jnp.asarray(radius, dtype=float).reshape(())

    @property
    def num_charts(self) -> int:
        return 4

    @property
    def reference_dimension(self) -> int:
        return 1

    @property
    def ambient_dimension(self) -> int:
        return 2

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        angle = 0.5 * jnp.pi * (chart_indices.astype(reference.dtype) + reference[..., 0])
        direction = jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
        return self.center + self.radius * direction

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        del chart_indices
        return jnp.broadcast_to(0.5 * jnp.pi * self.radius, reference.shape[:-1])


class _SphereBoundaryMap(AbstractBoundaryMap):
    center: Array
    radius: Array

    def __init__(self, center: Array, radius: Array):
        self.center = jnp.asarray(center, dtype=float).reshape((3,))
        self.radius = jnp.asarray(radius, dtype=float).reshape(())

    @property
    def num_charts(self) -> int:
        return 1

    @property
    def reference_dimension(self) -> int:
        return 2

    @property
    def ambient_dimension(self) -> int:
        return 3

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        del chart_indices
        azimuth = 2.0 * jnp.pi * reference[..., 0]
        vertical = 1.0 - 2.0 * reference[..., 1]
        radial = jnp.sqrt(jnp.maximum(1.0 - vertical * vertical, 0.0))
        direction = jnp.stack(
            (
                radial * jnp.cos(azimuth),
                radial * jnp.sin(azimuth),
                vertical,
            ),
            axis=-1,
        )
        return self.center + self.radius * direction

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        del chart_indices
        return jnp.broadcast_to(
            4.0 * jnp.pi * self.radius**2,
            reference.shape[:-1],
        )


class _BoxBoundaryMap(AbstractBoundaryMap):
    origins: Array
    first_axes: Array
    second_axes: Array
    jacobians: Array

    def __init__(self, center: Array, size: Array):
        center_ = jnp.asarray(center, dtype=float).reshape((3,))
        size_ = jnp.asarray(size, dtype=float).reshape((3,))
        half = 0.5 * size_
        hx, hy, hz = half
        dx, dy, dz = size_
        self.origins = center_ + jnp.asarray(
            [
                [-hx, -hy, -hz],
                [hx, -hy, -hz],
                [-hx, -hy, -hz],
                [-hx, hy, -hz],
                [-hx, -hy, -hz],
                [-hx, -hy, hz],
            ]
        )
        self.first_axes = jnp.asarray(
            [
                [0.0, dy, 0.0],
                [0.0, dy, 0.0],
                [dx, 0.0, 0.0],
                [dx, 0.0, 0.0],
                [dx, 0.0, 0.0],
                [dx, 0.0, 0.0],
            ]
        )
        self.second_axes = jnp.asarray(
            [
                [0.0, 0.0, dz],
                [0.0, 0.0, dz],
                [0.0, 0.0, dz],
                [0.0, 0.0, dz],
                [0.0, dy, 0.0],
                [0.0, dy, 0.0],
            ]
        )
        self.jacobians = jnp.asarray(
            [dy * dz, dy * dz, dx * dz, dx * dz, dx * dy, dx * dy]
        )

    @property
    def num_charts(self) -> int:
        return 6

    @property
    def reference_dimension(self) -> int:
        return 2

    @property
    def ambient_dimension(self) -> int:
        return 3

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        origins = self.origins[chart_indices]
        first_axes = self.first_axes[chart_indices]
        second_axes = self.second_axes[chart_indices]
        return (
            origins + reference[..., :1] * first_axes + reference[..., 1:2] * second_axes
        )

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        del reference
        return self.jacobians[chart_indices]


def circle_boundary_atlas(
    center: Array,
    radius: Array,
    /,
    *,
    source_id: str,
) -> BoundaryAtlas:
    return BoundaryAtlas(
        _CircleBoundaryMap(center, radius),
        source_entity_ids=jnp.zeros((4,), dtype=jnp.int32),
        source_id=source_id,
    )


def sphere_boundary_atlas(
    center: Array,
    radius: Array,
    /,
    *,
    source_id: str,
) -> BoundaryAtlas:
    return BoundaryAtlas(
        _SphereBoundaryMap(center, radius),
        source_entity_ids=jnp.asarray([0], dtype=jnp.int32),
        orientation=-jnp.ones((1,), dtype=float),
        source_id=source_id,
    )


def box_boundary_atlas(
    center: Array,
    size: Array,
    /,
    *,
    source_id: str,
) -> BoundaryAtlas:
    return BoundaryAtlas(
        _BoxBoundaryMap(center, size),
        orientation=jnp.asarray([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0]),
        physical_tags=("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"),
        source_entity_ids=jnp.arange(6, dtype=jnp.int32),
        source_id=source_id,
    )


BoundaryMap = AbstractBoundaryMap


__all__ = [
    "AbstractBoundaryMap",
    "TrimDomain",
    "BoundaryAtlas",
    "BoundaryFrame",
    "BoundaryMap",
]
