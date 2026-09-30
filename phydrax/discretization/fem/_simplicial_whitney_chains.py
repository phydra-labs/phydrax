#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
from math import comb, factorial
from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...exterior._chains import (
    AbstractChainIntegrationKernel,
    PreparedChainQuery,
    SegmentWeight,
)
from ...linalg import compound_matrix
from ...typing import AnyDim, Dim, Float, Int32, Size
from .._cell_de_rham import AbstractCellDeRhamComplex
from .._simplicial_locator import (
    CellLocationResult,
    CellLocationStatus,
    PreparedSimplicialCellLocator,
    SegmentLocationResult,
)
from .._topology import CellComplexTopology


class WhitneyCellDim(Dim):
    pass


class WhitneyLocalEntityDim(Dim):
    """Local simplex entities for one admitted form degree."""


@final
class _WhitneyDegreeRoutes(StrictModule):
    """Oriented global entity routes with one degree-specific shape contract."""

    __strict_contract__ = True

    indices: Int32[WhitneyCellDim, WhitneyLocalEntityDim]
    signs: Int32[WhitneyCellDim, WhitneyLocalEntityDim]
    cell_count: Size[WhitneyCellDim] = eqx.field(static=True)
    local_count: Size[WhitneyLocalEntityDim] = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    vertex_count: int = eqx.field(static=True)
    entity_count: int = eqx.field(static=True)

    def __init__(
        self,
        indices: np.ndarray,
        signs: np.ndarray,
        /,
        *,
        degree: int,
        vertex_count: int,
        entity_count: int,
    ) -> None:
        if vertex_count < 2 or not 0 <= degree < vertex_count or entity_count < 1:
            raise ValueError("Whitney route metadata has an invalid simplex degree.")
        local_count = comb(vertex_count, degree + 1)
        if indices.ndim != 2 or indices.shape[1] != local_count:
            raise ValueError("Whitney entity routes do not match the local degree.")
        if indices.dtype != np.dtype(np.int32) or signs.dtype != np.dtype(np.int32):
            raise TypeError("Whitney routes and orientation signs must be int32.")
        if signs.shape != indices.shape:
            raise ValueError("Whitney route indices and signs must have equal shape.")
        if np.any(indices < 0) or np.any(indices >= entity_count):
            raise ValueError("Whitney entity route lies outside its degree.")
        if np.any(np.abs(signs) != 1):
            raise ValueError("Whitney entity orientation must be plus or minus one.")
        self.indices = jnp.asarray(indices)
        self.signs = jnp.asarray(signs)
        self.cell_count = indices.shape[0]
        self.local_count = local_count
        self.degree = degree
        self.vertex_count = vertex_count
        self.entity_count = entity_count


def _whitney_two(barycentric: Array, gradients: Array, /) -> Array:
    vertex_count, ambient = gradients.shape[-2:]
    components = tuple(itertools.combinations(range(ambient), 2))
    forms = []
    for i, j, k in itertools.combinations(range(vertex_count), 3):
        terms = []
        for a, b in components:
            terms.append(
                2
                * (
                    barycentric[..., i]
                    * (
                        gradients[..., j, a] * gradients[..., k, b]
                        - gradients[..., j, b] * gradients[..., k, a]
                    )
                    - barycentric[..., j]
                    * (
                        gradients[..., i, a] * gradients[..., k, b]
                        - gradients[..., i, b] * gradients[..., k, a]
                    )
                    + barycentric[..., k]
                    * (
                        gradients[..., i, a] * gradients[..., j, b]
                        - gradients[..., i, b] * gradients[..., j, a]
                    )
                )
            )
        forms.append(jnp.stack(terms, axis=-1))
    return jnp.stack(forms, axis=-2)


def whitney_basis(barycentric: Array, gradients: Array, degree: int, /) -> Array:
    """Affine Whitney coefficients in increasing exterior-component order."""
    vertex_count, ambient = gradients.shape[-2:]
    if degree < 0 or degree >= vertex_count:
        raise ValueError("Whitney degree must lie within the simplex dimension.")
    if degree == 0:
        return barycentric[..., :, None]
    if degree == 1:
        return jnp.stack(
            [
                barycentric[..., i, None] * gradients[..., j, :]
                - barycentric[..., j, None] * gradients[..., i, :]
                for i, j in itertools.combinations(range(vertex_count), 2)
            ],
            axis=-2,
        )
    if degree == 2:
        return _whitney_two(barycentric, gradients)
    if 0 <= degree < vertex_count:
        forms = []
        for vertices in itertools.combinations(range(vertex_count), degree + 1):
            value = jnp.zeros(
                (
                    *barycentric.shape[:-1],
                    len(tuple(itertools.combinations(range(ambient), degree))),
                ),
                dtype=barycentric.dtype,
            )
            for omitted, vertex in enumerate(vertices):
                remaining = vertices[:omitted] + vertices[omitted + 1 :]
                minor = compound_matrix(
                    gradients[..., jnp.asarray(remaining), :], degree
                )[..., 0, :]
                value = value + (-1) ** omitted * barycentric[..., vertex, None] * minor
            forms.append(factorial(degree) * value)
        return jnp.stack(forms, axis=-2)
    raise ValueError("Whitney degree must lie within the simplex dimension.")


def _entity_support(
    topology: CellComplexTopology, /
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Recover simplex supports and orientations from sparse incidences."""
    supports = [tuple((i,) for i in range(topology.entities(0).count))]
    for incidence in topology.incidences:
        boundary = incidence.scipy_boundary().tocsc()
        previous = supports[-1]
        degree_support = []
        for entity in range(boundary.shape[1]):
            lower = boundary.indices[
                boundary.indptr[entity] : boundary.indptr[entity + 1]
            ]
            vertices = tuple(sorted({v for index in lower for v in previous[index]}))
            if len(vertices) != incidence.degree + 1:
                raise ValueError("Whitney reconstruction requires simplicial incidence.")
            degree_support.append(vertices)
        supports.append(tuple(degree_support))
    return tuple(supports)


def _entity_orientations(
    topology: CellComplexTopology, supports: tuple[tuple[tuple[int, ...], ...], ...], /
) -> tuple[np.ndarray, ...]:
    orientations = [np.ones(topology.entities(0).count, dtype=np.int32)]
    for degree in range(1, len(supports)):
        boundary = topology.incidences[degree - 1].scipy_boundary().tocsc()
        lookup = {vertices: i for i, vertices in enumerate(supports[degree - 1])}
        orientation = np.empty(len(supports[degree]), dtype=np.int32)
        for index, vertices in enumerate(supports[degree]):
            lower = lookup[vertices[1:]]
            orientation[index] = int(boundary[lower, index]) * orientations[-1][lower]
            if abs(orientation[index]) != 1:
                raise ValueError("Whitney simplex incidence must have unit orientation.")
        orientations.append(orientation)
    return tuple(orientations)


def _simplex_routes(
    cells: np.ndarray,
    supports: tuple[tuple[int, ...], ...],
    orientation: np.ndarray,
    degree: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {vertices: i for i, vertices in enumerate(supports)}
    local = tuple(itertools.combinations(range(cells.shape[1]), degree + 1))
    ids = np.empty((cells.shape[0], len(local)), dtype=np.int32)
    factors = np.empty_like(ids)
    for cell_index, cell in enumerate(cells):
        for slot, subset in enumerate(local):
            oriented = tuple(int(cell[i]) for i in subset)
            canonical = tuple(sorted(oriented))
            if canonical not in lookup:
                raise ValueError("Locator simplex is absent from the chain topology.")
            entity = lookup[canonical]
            parity = sum(
                oriented[i] > oriented[j]
                for i in range(len(oriented))
                for j in range(i + 1, len(oriented))
            )
            ids[cell_index, slot] = entity
            factors[cell_index, slot] = (-1 if parity % 2 else 1) * orientation[entity]
    return ids, factors


@final
class SimplicialWhitneyKernel(AbstractChainIntegrationKernel):
    """Whitney reconstruction and exact affine facet-split line integrals."""

    __strict_contract__ = True

    complex: AbstractCellDeRhamComplex | CellComplexTopology
    locator: PreparedSimplicialCellLocator
    cell_routes: tuple[_WhitneyDegreeRoutes, ...]
    gradients: Float[WhitneyCellDim, AnyDim, AnyDim]
    dof_counts: tuple[int, ...] = eqx.field(static=True)
    dof_offsets: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex: AbstractCellDeRhamComplex | CellComplexTopology,
        locator: PreparedSimplicialCellLocator,
        /,
    ) -> None:
        if not isinstance(complex, (AbstractCellDeRhamComplex, CellComplexTopology)):
            raise TypeError("complex must be a cell de Rham complex or cell topology.")
        if not isinstance(locator, PreparedSimplicialCellLocator):
            raise TypeError("locator must be a prepared simplicial locator.")
        if locator.cell_map.coordinate_element.degree != 1:
            raise ValueError(
                "Whitney chain integration requires affine simplex geometry."
            )
        topology = (
            complex if isinstance(complex, CellComplexTopology) else complex.topology
        )
        if (
            topology.dimension != locator.dimension
            or topology.entities(0).count != locator.coordinate_count
        ):
            raise ValueError("Whitney topology and locator dimensions disagree.")
        supports = _entity_support(topology)
        cells = np.asarray(locator.cells, dtype=np.int32)
        orientations = _entity_orientations(topology, supports)
        prepared_routes = tuple(
            _simplex_routes(cells, supports[degree], orientations[degree], degree)
            for degree in range(topology.dimension + 1)
        )
        grouped_routes = tuple(
            _WhitneyDegreeRoutes(
                indices,
                signs,
                degree=degree,
                vertex_count=cells.shape[1],
                entity_count=topology.entities(degree).count,
            )
            for degree, (indices, signs) in enumerate(prepared_routes)
        )
        counts = tuple(entity.count for entity in topology.entity_sets)
        if (
            isinstance(complex, AbstractCellDeRhamComplex)
            and complex.cell_counts != counts
        ):
            raise ValueError(
                "Whitney reconstruction requires one scalar coefficient per simplex."
            )
        gradients = locator.affine_gradients()
        self.complex = complex
        self.locator = locator
        self.gradients = gradients
        self.cell_routes = grouped_routes
        self.dof_counts = counts
        self.dof_offsets = tuple((0, count) for count in counts)
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "simplicial-whitney-kernel",
                "topology": topology.topology_id,
                "locator": locator.locator_id,
            }
        )

    def evaluate(
        self,
        points: ArrayLike,
        degree: int,
        /,
        *,
        proxy: Literal["components", "flux"] = "components",
        location: CellLocationResult | None = None,
    ) -> PreparedChainQuery:
        if degree < 0 or degree >= len(self.cell_routes):
            raise ValueError("Whitney degree must lie within the simplex dimension.")
        if location is None:
            location = self.locator.locate(points)
        elif (
            location.locator_id != self.locator.locator_id
            or location.cell_ids.shape != (jnp.asarray(points).shape[0],)
        ):
            raise ValueError("Prepared point location does not match this Whitney query.")
        cells = jnp.maximum(location.cell_ids, 0)
        coefficients = (
            location.barycentric[..., None]
            if degree == 0
            else whitney_basis(location.barycentric, self.gradients[cells], degree)
        )
        if proxy == "flux":
            if degree != 2 or self.locator.coordinates.shape[1] != 3:
                raise ValueError(
                    "Whitney flux proxy requires degree two in three dimensions."
                )
            coefficients = coefficients[..., jnp.asarray((2, 1, 0))] * jnp.asarray(
                (1, -1, 1)
            )
        elif proxy != "components":
            raise ValueError("Unknown Whitney reconstruction proxy.")
        coefficients = coefficients * self.cell_routes[degree].signs[cells, :, None]
        valid = jnp.broadcast_to(location.inside[:, None], coefficients.shape[:2])
        overflow = location.status == int(CellLocationStatus.RESOURCE_EXCEEDED)
        return PreparedChainQuery(
            self.cell_routes[degree].indices[cells],
            coefficients,
            valid,
            location.successful,
            overflow,
            dof_count=self.dof_counts[degree],
            degree=degree,
            kernel_id=self.kernel_id,
        )

    def _point_query(self, location: CellLocationResult, /) -> PreparedChainQuery:
        cells = jnp.maximum(location.cell_ids, 0)
        indices = self.cell_routes[0].indices[cells]
        valid = jnp.broadcast_to(location.inside[:, None], indices.shape)
        overflow = location.status == int(CellLocationStatus.RESOURCE_EXCEEDED)
        return PreparedChainQuery(
            indices,
            location.barycentric,
            valid,
            location.successful,
            overflow,
            dof_count=self.dof_counts[0],
            degree=0,
            kernel_id=self.kernel_id,
        )

    def integrate_points(self, points: ArrayLike, /) -> PreparedChainQuery:
        return self._point_query(self.locator.locate(points))

    def prepare_trajectory(
        self, start: ArrayLike, end: ArrayLike, /, *, maximum_segments: int | None = None
    ) -> tuple[PreparedChainQuery, PreparedChainQuery, PreparedChainQuery]:
        """Prepare endpoint charge and line-current routes with one facet traversal."""
        left = jnp.asarray(start)
        right = jnp.asarray(end, dtype=left.dtype)
        segment = self.locator.locate_segment(
            left, right, maximum_segments=maximum_segments
        )
        route = self._segment_query(left, right, segment, "uniform", None)
        return self._point_query(segment.start), self._point_query(segment.end), route

    def integrate_segments(
        self,
        start: ArrayLike,
        end: ArrayLike,
        /,
        *,
        weight: SegmentWeight = "uniform",
        phase_rate: ArrayLike | None = None,
        maximum_segments: int | None = None,
    ) -> PreparedChainQuery:
        left = jnp.asarray(start)
        right = jnp.asarray(end, dtype=left.dtype)
        segment = self.locator.locate_segment(
            left, right, maximum_segments=maximum_segments
        )
        return self._segment_query(left, right, segment, weight, phase_rate)

    def _segment_query(
        self,
        left: Array,
        right: Array,
        segment: SegmentLocationResult,
        weight: SegmentWeight,
        phase_rate: ArrayLike | None,
        /,
    ) -> PreparedChainQuery:
        cells = jnp.maximum(segment.cell_ids, 0)
        gradients = self.gradients[cells]
        origins = self.locator.coordinates[self.locator.cells[cells, 0]]
        a = jnp.sum(
            gradients * (left[:, None, None, :] - origins[:, :, None, :]), axis=-1
        )
        a = a.at[..., 0].add(1)
        change = jnp.sum(gradients * (right - left)[:, None, None, :], axis=-1)
        t0, t1 = segment.intervals[..., 0], segment.intervals[..., 1]
        integrals = jnp.stack(
            [
                a[..., i] * change[..., j] - a[..., j] * change[..., i]
                for i, j in itertools.combinations(range(self.locator.cells.shape[1]), 2)
            ],
            axis=-1,
        )
        if weight == "uniform":
            if phase_rate is not None:
                raise ValueError(
                    "phase_rate is only defined for phase-weighted integration."
                )
            moment = t1 - t0
        elif weight == "phase":
            if phase_rate is None:
                raise ValueError("Phase integration requires phase_rate.")
            rate = jnp.asarray(phase_rate, dtype=left.dtype)
            rate = jnp.broadcast_to(rate, (left.shape[0],))[:, None]
            width = t1 - t0
            moment = (
                width
                * jnp.exp(1j * rate * (t0 + t1) / 2)
                * jnp.sinc(rate * width / (2 * jnp.pi))
            )
        else:
            raise ValueError("Unknown segment weight.")
        coefficients = integrals * moment[..., None] * self.cell_routes[1].signs[cells]
        indices = self.cell_routes[1].indices[cells]
        valid = jnp.broadcast_to(segment.valid[..., None], indices.shape)
        return PreparedChainQuery(
            indices.reshape((left.shape[0], -1)),
            coefficients.reshape((left.shape[0], -1)),
            valid.reshape((left.shape[0], -1)),
            segment.successful,
            segment.overflow,
            dof_count=self.dof_counts[1],
            degree=1,
            kernel_id=self.kernel_id,
        )


__all__ = ["SimplicialWhitneyKernel"]
