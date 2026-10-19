#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Support geometry and location status shared by discrete field views.

Structured reconstructions cover a tensor box; simplicial boundaries retain
their affine distance provider; mapped cells retain the entire coordinate
source. Explicit support must carry whole-domain binding or boundary evidence.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from enum import IntEnum
from fractions import Fraction
from math import isfinite
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.discretization.fem._reference_types import FiniteElementSpec

from .. import geometry as imported_geometry
from .._strict import StrictModule
from ..geometry import simplicial as imported_geometry_simplicial
from ..geometry._atlas import AbstractBoundaryMap, BoundaryAtlas
from ..geometry._capabilities import GeometryCapability
from ..geometry._certificate import (
    DistanceSemantics,
    FieldCertificate,
    FieldRegularity,
    SignReliability,
    ZeroSetAccuracy,
)
from ..geometry._contracts import (
    CompiledGeometry,
    GeometryKernel,
    GeometryKind,
    GeometrySource,
    GeometryTolerance,
)
from ..geometry.analytic import _primitives as imported_geometry_analytic__primitives
from ..geometry.design._schema import (
    DesignState,
    ParameterBinding,
    ParameterCollector,
    ParameterId,
)
from ..geometry.simplicial import (
    _io as imported_geometry_simplicial__io,
    _regions as imported_geometry_simplicial__regions,
    _support_region as imported_geometry_simplicial__support_region,
)
from ..typing import checked, PRNGKey
from ..units import METER, UnitDefinition
from ._cell_complex import (
    IntervalConnectivity,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ._cell_geometry import (
    cell_geometry_preparation_bounds,
    cell_geometry_preparation_query_upper,
    owned_coordinate_preparation,
)
from ._cell_mesh import SimplicialConnectivity
from ._coordinate_budget import CoordinateEnclosureBudget
from ._coordinate_enclosure import (
    coordinate_chart_restriction_estimates,
    coordinate_estimate_bank,
    coordinate_estimate_equal,
    coordinate_evaluate_estimate,
    coordinate_expressions,
    coordinate_map_expression_estimates,
    coordinate_polynomials_estimate,
    Expression,
    expression_sequence_equal,
    restrict_chart_expressions,
)
from ._hexahedral import HexahedralConnectivity
from ._mapped_locator import PreparedMappedCellLocator, reference_domain
from ._polyhedral_locator import PreparedPolyhedralCellLocator
from ._reference_cell import reference_cell_topology, ReferenceCellTopology
from ._simplicial_locator import (
    AbstractCellLocator,
    cell_locator_preparation_bounds,
    cell_map_vertices,
    CellLocationResult,
    CellLocationStatus,
    PreparedSimplicialCellLocator,
)
from ._views import FieldQueryStatus
from .fem._cell_map import PreparedFiniteElementCellMap
from .fem._reference_operator import (
    facet_corner_parameters,
    reference_facet_embedding,
    reference_facet_shape,
)


if TYPE_CHECKING:
    from ..geometry._sampling import RejectionSamplingPlan, SamplingResult


def cell_location_status(status: Array, /) -> Array:
    """Map `CellLocationStatus` codes onto field query statuses."""
    return jnp.where(
        status == int(CellLocationStatus.LOCATED),
        int(FieldQueryStatus.VALID),
        jnp.where(
            status == int(CellLocationStatus.OUTSIDE),
            int(FieldQueryStatus.OUTSIDE_SUPPORT),
            jnp.where(
                status == int(CellLocationStatus.NONFINITE),
                int(FieldQueryStatus.NONFINITE),
                int(FieldQueryStatus.LOCATION_FAILED),
            ),
        ),
    ).astype(jnp.int32)


class MappedSupportQueryStatus(IntEnum):
    """Location status values plus revision-bound source invalidation."""

    SOURCE_REVISION_MISMATCH = 6


class MappedMeshSupportQueryResult(StrictModule):
    inside: Array
    status: Array
    successful: Array
    source_current: Array
    location: CellLocationResult
    source_binding_id: str = eqx.field(static=True)


class MappedCellBoundaryMap(AbstractBoundaryMap):
    """Reference facets composed with the coordinate source."""

    cell_map: PreparedFiniteElementCellMap
    coordinates: Array
    chart_cells: Array
    chart_facets: Array
    facet_axes: Array
    simplicial_facets: Array
    facet_normals: Array
    facet_offsets: Array
    exterior_facets: Array

    @property
    def num_charts(self) -> int:
        return self.chart_cells.shape[0]

    @property
    def reference_dimension(self) -> int:
        return self.cell_map.reference_dimension - 1

    @property
    def ambient_dimension(self) -> int:
        return self.cell_map.ambient_dimension

    def _reference(
        self, chart_indices: Array, reference: Array, /
    ) -> tuple[Array, Array, Array]:
        indices = chart_indices.reshape((-1,))
        parameters = reference.reshape((-1, self.reference_dimension))
        local_facets = self.chart_facets[indices]
        axes = self.facet_axes[local_facets]
        facet_parameters = parameters
        if self.reference_dimension:
            simplex = self.simplicial_facets[local_facets]
            dimension = self.reference_dimension
            transformed = []
            derivatives = []
            for axis in range(dimension):
                prefix = jnp.prod(1.0 - parameters[:, :axis], axis=1)
                transformed.append(parameters[:, axis] * prefix)
                row = []
                for column in range(dimension):
                    if column == axis:
                        row.append(prefix)
                    elif column < axis:
                        others = tuple(index for index in range(axis) if index != column)
                        row.append(
                            -parameters[:, axis]
                            * jnp.prod(
                                1.0 - parameters[:, jnp.asarray(others, dtype=jnp.int32)],
                                axis=1,
                            )
                        )
                    else:
                        row.append(jnp.zeros_like(prefix))
                derivatives.append(jnp.stack(row, axis=-1))
            facet_parameters = jnp.where(
                simplex[:, None], jnp.stack(transformed, axis=-1), parameters
            )
            derivative = jnp.where(
                simplex[:, None, None],
                jnp.stack(derivatives, axis=1),
                jnp.eye(dimension, dtype=parameters.dtype)[None, :, :],
            )
            axes = axes @ derivative
        kind = self.cell_map.coordinate_element.cell_kind
        facets = reference_cell_topology(kind).entities[
            self.cell_map.reference_dimension - 1
        ]
        points = jnp.zeros(
            (indices.size, self.cell_map.reference_dimension), dtype=parameters.dtype
        )
        for facet in range(len(facets)):
            embedded, _ = reference_facet_embedding(kind, facet, facet_parameters)
            points = jnp.where((local_facets == facet)[:, None], embedded, points)
        return self.chart_cells[indices], points, axes

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        cells, points, _ = self._reference(chart_indices, reference)
        evaluation = self.cell_map.evaluate(self.coordinates, cells, points)
        return evaluation.physical_points.reshape(
            (*chart_indices.shape, self.ambient_dimension)
        )

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        cells, points, axes = self._reference(chart_indices, reference)
        if self.reference_dimension == 0:
            return jnp.ones(chart_indices.shape, dtype=self.coordinates.dtype)
        evaluation = self.cell_map.evaluate(self.coordinates, cells, points)
        tangent = evaluation.jacobian @ axes
        metric = jnp.swapaxes(tangent, -1, -2) @ tangent
        density = jnp.sqrt(jnp.maximum(jnp.linalg.det(metric), 0.0))
        return density.reshape(chart_indices.shape)


def _facet_span(corners: np.ndarray, dimension: int, /) -> np.ndarray:
    """Independent directions in the given global facet-vertex order."""
    directions: list[np.ndarray] = []
    for corner in corners[1:]:
        candidate = corner - corners[0]
        if np.linalg.matrix_rank(np.stack((*directions, candidate), axis=1)) > len(
            directions
        ):
            directions.append(candidate)
            if len(directions) == dimension - 1:
                break
    if len(directions) != dimension - 1:
        raise ValueError(
            "mapped_support.reference_facets must have codimension one; got incompatible corner span"
        )
    return np.stack(directions, axis=1) if directions else np.empty((dimension, 0))


def _mapped_facet_occurrences(
    cell_map: PreparedFiniteElementCellMap, /
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...], dict[int, list[tuple[int, int]]]]:
    """Return facet rows and their cell-local occurrences."""
    mesh = cell_map.mesh
    kind = cell_map.coordinate_element.cell_kind
    topology = reference_cell_topology(kind)
    facets = topology.entities[topology.dimension - 1]
    cells = cell_map_vertices(cell_map)
    connectivity = mesh.connectivity
    match connectivity:
        case SimplicialConnectivity():
            global_facets = tuple(
                tuple(int(value) for value in row)
                for row in np.asarray(connectivity.entities[topology.dimension - 1])
            )
        case IntervalConnectivity() if topology.dimension == 1:
            global_facets = tuple((index,) for index in range(mesh.coordinates.shape[0]))
        case PolygonalConnectivity() if topology.dimension == 2:
            global_facets = tuple(
                tuple(int(value) for value in row)
                for row in np.asarray(connectivity.edges)
            )
        case TetrahedralConnectivity() | HexahedralConnectivity() if (
            topology.dimension == 3
        ):
            global_facets = tuple(
                tuple(int(value) for value in row)
                for row in np.asarray(connectivity.faces)
            )
        case PolyhedralConnectivity() if topology.dimension == 3:
            offsets = np.asarray(connectivity.face_vertex_offsets)
            values = np.asarray(connectivity.face_vertex_values)
            global_facets = tuple(
                tuple(int(value) for value in values[start:stop])
                for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
            )
        case _:
            raise TypeError(
                "mapped_support.connectivity must match the reference topology; got unsupported connectivity"
            )
    facet_slots = {
        tuple(sorted(vertices)): slot for slot, vertices in enumerate(global_facets)
    }
    occurrences: dict[int, list[tuple[int, int]]] = {}
    for cell, vertices in enumerate(cells):
        for local_facet, facet_vertices in enumerate(facets):
            key = tuple(sorted(int(vertices[index]) for index in facet_vertices))
            slot = facet_slots[key]
            occurrences.setdefault(slot, []).append((cell, local_facet))
    return cells, global_facets, occurrences


def _mapped_support_preparation_bounds(
    cell_map: PreparedFiniteElementCellMap, coordinates: np.ndarray, /
) -> tuple[int, int, int]:
    """Bound authored source banks, full facet restrictions and trace equality."""
    geometry = cell_map.coordinate_spec
    work, memory = cell_geometry_preparation_bounds(geometry)
    queries = cell_geometry_preparation_query_upper(geometry)
    cells, global_facets, occurrences = _mapped_facet_occurrences(cell_map)
    interior = tuple(
        (slot, entries) for slot, entries in occurrences.items() if len(entries) == 2
    )
    if not interior:
        return work, memory, queries
    source = geometry.exact_source
    periodic = geometry.periodic_source
    coefficient_bits = (
        periodic.preparation_bounds()[2]
        if periodic is not None
        else max(source.maximum_bits, 1075)
        if source is not None
        else None
    )
    routes = np.asarray(cell_map.coordinate_dofs)
    local = coordinates[routes[interior[0][1][0][0]]]
    element = cell_map.coordinate_element
    preparation = coordinate_polynomials_estimate(
        element, local, coefficient_bits_upper=coefficient_bits
    )
    expressions = coordinate_map_expression_estimates(
        element, local, coefficient_bits_upper=coefficient_bits
    )
    if preparation is None or expressions is None:
        raise ValueError(
            "mapped_support.coordinate_expressions must retain source traces; got an unsupported source expression"
        )
    touched = {cell for _, entries in interior for cell, _ in entries}
    work += len(touched) * preparation.work_upper
    memory += len(touched) * preparation.bytes_upper
    topology = reference_cell_topology(element.cell_kind)
    for slot, entries in interior:
        traces = []
        for cell, _ in entries:
            positions = {
                int(vertex): position for position, vertex in enumerate(cells[cell])
            }
            corners = np.asarray(
                tuple(
                    topology.vertices[positions[vertex]] for vertex in global_facets[slot]
                )
            )
            axes = _facet_span(corners, topology.dimension)
            target_kind = (
                "interval"
                if axes.shape[1] == 1
                else "triangle"
                if len(corners) == 3
                else "quadrilateral"
            )
            trace = coordinate_chart_restriction_estimates(
                expressions, element.cell_kind, target_kind, corners[0], axes
            )
            bank = coordinate_estimate_bank(trace)
            work += bank.work_upper
            memory += bank.bytes_upper
            traces.append(trace)
        for first, second in zip(traces[0], traces[1], strict=True):
            equality = coordinate_estimate_equal(first, second)
            work += equality.work_upper
            memory += equality.bytes_upper
    return work, memory, queries


@contextmanager
def _owned_mapped_support_preparation(
    cell_map: PreparedFiniteElementCellMap,
    coordinates: Array,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
    include_enclosure: bool,
) -> Iterator[CoordinateEnclosureBudget]:
    """Own the host support circuit or reuse its explicit request ledger."""
    if budget is not None:
        yield budget
        return
    host = np.asarray(coordinates)
    work, memory, queries = _mapped_support_preparation_bounds(cell_map, host)
    if include_enclosure:
        enclosure_work, enclosure_memory, enclosure_queries = (
            cell_locator_preparation_bounds(
                cell_map, host, reference_domain(cell_map.coordinate_element.cell_kind)
            )
        )
        work += enclosure_work
        memory += enclosure_memory
        queries += enclosure_queries
    with owned_coordinate_preparation(
        work, memory, maximum_geometry_queries=queries
    ) as budget:
        yield budget


def _affine_support_preparation_bounds(
    cell_map: PreparedFiniteElementCellMap, coordinates: np.ndarray, /
) -> tuple[int, int, int]:
    """Bound source expansion and every reference-corner evaluation."""
    geometry = cell_map.coordinate_spec
    work, memory = cell_geometry_preparation_bounds(geometry)
    queries = cell_geometry_preparation_query_upper(geometry)
    source = geometry.exact_source
    periodic = geometry.periodic_source
    coefficient_bits = (
        periodic.preparation_bounds()[2]
        if periodic is not None
        else max(source.maximum_bits, 1075)
        if source is not None
        else None
    )
    routes = np.asarray(cell_map.coordinate_dofs)
    if routes.shape[0]:
        local = coordinates[routes[0]]
        element = cell_map.coordinate_element
        preparation = coordinate_polynomials_estimate(
            element, local, coefficient_bits_upper=coefficient_bits
        )
        expressions = coordinate_map_expression_estimates(
            element, local, coefficient_bits_upper=coefficient_bits
        )
        if preparation is None or expressions is None:
            raise ValueError(
                "affine_support.coordinate_expressions must retain source coordinates; got an unsupported source expression"
            )
        work += routes.shape[0] * preparation.work_upper
        memory += routes.shape[0] * preparation.bytes_upper
        for corner in reference_cell_topology(element.cell_kind).vertices:
            point = tuple(Fraction(value) for value in corner)
            for value in expressions:
                evaluation = coordinate_evaluate_estimate(value, point)
                work += routes.shape[0] * evaluation.work_upper
                memory += routes.shape[0] * evaluation.bytes_upper
    return work, memory, queries


@contextmanager
def _owned_affine_support_preparation(
    cell_map: PreparedFiniteElementCellMap,
    coordinates: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> Iterator[CoordinateEnclosureBudget]:
    """Own the affine source proof without changing projection-map limits."""
    if budget is not None:
        yield budget
        return
    work, memory, queries = _affine_support_preparation_bounds(cell_map, coordinates)
    with owned_coordinate_preparation(
        work, memory, maximum_geometry_queries=queries
    ) as budget:
        yield budget


class _MappedBoundaryProof(StrictModule):
    """Exterior source facets after full coordinate-trace agreement."""

    exterior: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    ids: np.ndarray


def _prepare_mapped_boundary_proof(
    locator: PreparedMappedCellLocator,
    topology: ReferenceCellTopology,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _MappedBoundaryProof:
    cell_map = locator.cell_map
    mesh = cell_map.mesh
    kind = cell_map.coordinate_element.cell_kind
    cells, global_facets, occurrences = _mapped_facet_occurrences(cell_map)
    # Interior facets disappear only after whole source traces agree.
    source_polynomials: dict[int, tuple[Expression, ...] | None] = {}
    routes = np.asarray(cell_map.coordinate_dofs)
    coordinates = cell_map.source_coordinates(locator.coordinates, budget=budget)

    def trace(cell: int, slot: int) -> tuple[Expression, ...]:
        if cell not in source_polynomials:
            source_polynomials[cell] = coordinate_expressions(
                cell_map.coordinate_element,
                tuple(coordinates[index] for index in routes[cell]),
                budget=budget,
            )
        local_positions = {
            int(vertex): position for position, vertex in enumerate(cells[cell])
        }
        corners = np.asarray(
            tuple(
                topology.vertices[local_positions[vertex]]
                for vertex in global_facets[slot]
            )
        )
        origin = corners[0]
        axes = _facet_span(corners, topology.dimension)
        polynomials = source_polynomials[cell]
        if polynomials is None:
            raise ValueError(
                "mapped_support.coordinate_expressions must retain source traces; got an unsupported source expression"
            )
        target_kind = (
            "interval"
            if axes.shape[1] == 1
            else "triangle"
            if len(corners) == 3
            else "quadrilateral"
        )
        return restrict_chart_expressions(
            polynomials, kind, target_kind, origin, axes, budget=budget
        )

    for slot, entries in occurrences.items():
        if len(entries) == 2 and not expression_sequence_equal(
            trace(entries[0][0], slot), trace(entries[1][0], slot), budget=budget
        ):
            raise ValueError(
                "mapped_support.facet_traces must match full coordinate-source traces; got nonmatching facet traces"
            )
        if len(entries) > 2:
            raise ValueError(
                "mapped_support.facet_topology must be manifold; got more than two facet occurrences"
            )
    facet_ids = np.asarray(mesh.entity_set(topology.dimension - 1).entity_ids)
    exterior = tuple(
        sorted(
            (slot, entries[0][0], entries[0][1])
            for slot, entries in occurrences.items()
            if len(entries) == 1
        )
    )
    if not exterior:
        raise ValueError(
            "mapped_support.exterior_facets must be nonempty; got no represented exterior boundary"
        )
    ids = np.asarray([facet_ids[slot] for slot, _, _ in exterior], dtype=np.int64)
    if np.any(ids > np.iinfo(np.int32).max):
        raise ValueError(
            "mapped_support.facet_ids must fit represented int32 capacity; got a larger source facet ID"
        )
    return _MappedBoundaryProof(exterior, ids)


def _mapped_mesh_boundary_atlas(
    locator: PreparedMappedCellLocator,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> BoundaryAtlas:
    """Compile oriented exterior reference facets with mesh identities."""
    cell_map = locator.cell_map
    kind = cell_map.coordinate_element.cell_kind
    topology = reference_cell_topology(kind)
    facets = topology.entities[topology.dimension - 1]
    proof = _prepare_mapped_boundary_proof(locator, topology, budget=budget)
    exterior, ids = proof.exterior, proof.ids
    facet_axes = np.empty((len(facets), topology.dimension, topology.dimension - 1))
    signs = np.empty((len(facets),))
    facet_normals = np.empty((len(facets), topology.dimension))
    facet_offsets = np.empty((len(facets),))
    exterior_facets = np.zeros((cell_map.cell_count, len(facets)), dtype=np.bool_)
    for _, cell, facet in exterior:
        exterior_facets[cell, facet] = True
    for index, facet in enumerate(facets):
        corners = np.asarray(tuple(topology.vertices[vertex] for vertex in facet))
        parameter = np.concatenate(
            (np.zeros((1, topology.dimension - 1)), np.eye(topology.dimension - 1)),
            axis=0,
        )
        # Unit-axis parameters select reference corners without device dispatch.
        corner_parameters = facet_corner_parameters(reference_facet_shape(kind, index))
        selected = tuple(
            np.flatnonzero(np.all(corner_parameters == point, axis=1))[0]
            for point in parameter
        )
        embedded = corners[np.asarray(selected, dtype=np.int64)]
        facet_axes[index] = (embedded[1:] - embedded[:1]).T
        raw_normal = np.asarray(
            tuple(
                (-1.0) ** axis * np.linalg.det(np.delete(facet_axes[index], axis, axis=0))
                for axis in range(topology.dimension)
            )
        )
        outward = corners.mean(axis=0) - np.asarray(topology.vertices).mean(axis=0)
        signs[index] = 1.0 if raw_normal @ outward > 0.0 else -1.0
        # Unnormalized dyadic planes retain the represented reference zero set;
        # their margin is not a Euclidean distance.
        facet_normals[index] = signs[index] * raw_normal
        facet_offsets[index] = facet_normals[index] @ corners[0]
    mapping = MappedCellBoundaryMap(
        cell_map,
        locator.coordinates,
        jnp.asarray([cell for _, cell, _ in exterior], dtype=jnp.int32),
        jnp.asarray([facet for _, _, facet in exterior], dtype=jnp.int32),
        jnp.asarray(facet_axes),
        jnp.asarray([len(facet) == topology.dimension for facet in facets]),
        jnp.asarray(facet_normals),
        jnp.asarray(facet_offsets),
        jnp.asarray(exterior_facets),
    )
    return BoundaryAtlas(
        mapping,
        source_entity_ids=jnp.asarray(ids, dtype=jnp.int32),
        source_id=locator.source_binding_id,
        orientation=jnp.asarray([signs[facet] for _, _, facet in exterior]),
    )


class MappedMeshSupportSource(GeometrySource):
    """Union of prepared coordinate-map images, not corner polytopes."""

    locator: PreparedMappedCellLocator
    support_id: str = eqx.field(static=True)

    @checked
    def compile(
        self,
        *,
        coordinate_unit: UnitDefinition,
        tolerance: GeometryTolerance = GeometryTolerance(),
        budget: CoordinateEnclosureBudget | None = None,
    ) -> CompiledGeometry:
        """Compile whole mapped support using one bounded host coordinate request."""
        context = ParameterCollector(coordinate_unit)
        kernel = self._compile(context, budget=budget)
        _, state = context.finish()
        return CompiledGeometry(kernel, state, tolerance=tolerance)

    def _compile(
        self,
        context: ParameterCollector,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> GeometryKernel:
        # Pure geometry compositions call the polymorphic hook directly.
        # This source owns its host request when that hook has no caller ledger.
        with _owned_mapped_support_preparation(
            self.locator.cell_map,
            self.locator.coordinates,
            budget=budget,
            include_enclosure=False,
        ) as budget:
            coordinates = context.bind(
                ParameterId(f"mapped-cell-support:{self.support_id}", "coordinates"),
                self.locator.coordinates,
                role="position",
                unit=context.coordinate_unit,
                trainable=False,
            )
            return MappedMeshSupportKernel(
                self.locator,
                coordinates,
                _mapped_mesh_boundary_atlas(self.locator, budget=budget),
            )


class MappedMeshSupportKernel(GeometryKernel):
    locator: PreparedMappedCellLocator
    coordinates: ParameterBinding = eqx.field(static=True)
    atlas: BoundaryAtlas

    @property
    def ambient_dimension(self) -> int:
        return self.locator.cell_map.ambient_dimension

    @property
    def intrinsic_dimension(self) -> int:
        return self.locator.cell_map.reference_dimension

    @property
    def kind(self) -> GeometryKind:
        return (
            GeometryKind.REGION
            if self.ambient_dimension == self.intrinsic_dimension
            else GeometryKind.MANIFOLD
        )

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return frozenset(
            {GeometryCapability.REGION_QUERY, GeometryCapability.BOUNDARY_ATLAS}
        )

    @property
    def field_certificate(self) -> FieldCertificate:
        # Only retained exterior source facets contribute to the zero set.
        # The inverse remains bounded/tolerance-based, and reference-plane
        # magnitudes establish neither physical distance nor metric bounds.
        return FieldCertificate(
            ZeroSetAccuracy.APPROXIMATE,
            SignReliability.LOCAL,
            DistanceSemantics.LEVEL_SET,
            FieldRegularity.NONSMOOTH,
            None,
            "resolved_inverse_queries",
            False,
            ("whole_coordinate_source_union", "bounded_inverse_membership"),
            topology_identity=self.locator.cell_map.topology_id,
        )

    def _source_current(self, state: DesignState, /) -> Array:
        return jnp.all(self.coordinates.read(state) == self.locator.coordinates)

    def query(self, state: DesignState, points: Array, /) -> MappedMeshSupportQueryResult:
        values = jnp.asarray(points, dtype=self.locator.coordinates.dtype)
        if values.ndim < 1 or values.shape[-1] != self.ambient_dimension:
            raise ValueError(
                "MappedMeshSupportKernel.points must match the ambient dimension; got incompatible point shape"
            )
        leading = values.shape[:-1]
        location = self.locator.locate(values.reshape((-1, self.ambient_dimension)))
        current = self._source_current(state)
        status = jnp.where(
            current,
            location.status,
            int(MappedSupportQueryStatus.SOURCE_REVISION_MISMATCH),
        )
        resolved = (status == int(CellLocationStatus.LOCATED)) | (
            status == int(CellLocationStatus.OUTSIDE)
        )
        return MappedMeshSupportQueryResult(
            (location.inside & current).reshape(leading),
            status.reshape(leading),
            resolved.reshape(leading),
            jnp.broadcast_to(current, leading),
            location,
            self.locator.source_binding_id,
        )

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return self.query(state, points).inside

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        query = self.query(state, points)
        location = query.location
        mapping = self.atlas.mapping
        assert isinstance(mapping, MappedCellBoundaryMap)
        candidates = location.candidate_cells
        masks = (
            mapping.exterior_facets[jnp.maximum(candidates, 0)]
            & (candidates >= 0)[..., None]
        )
        margins = (
            mapping.facet_offsets[None, None, :]
            - location.candidate_reference @ mapping.facet_normals.T
        )
        margin = jnp.min(jnp.where(masks, margins, jnp.inf), axis=(1, 2))
        # A containing cell with no exterior facets is strictly interior.
        # Ignore such cells while other containing cells supply exterior
        # facets, so true exterior vertices remain zero even when also owned
        # by a cell with no exterior face.
        margin = jnp.where(jnp.isfinite(margin), margin, 1.0)
        field = jnp.where(query.inside.reshape((-1,)), -margin, 1.0)
        return jnp.where(query.successful, field.reshape(query.inside.shape), jnp.nan)

    def bounds(self, state: DesignState, /) -> Array:
        bounds = jnp.stack(
            (
                jnp.min(self.locator.cell_lower, axis=0),
                jnp.max(self.locator.cell_upper, axis=0),
            )
        )
        return jnp.where(self._source_current(state), bounds, jnp.nan)

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        raise NotImplementedError(
            "MappedMeshSupportKernel.boundary_normal must use a kernel providing "
            f"physical boundary normals; got {type(self).__name__}"
        )

    def measure(self, state: DesignState, /) -> Array:
        raise NotImplementedError(
            "MappedMeshSupportKernel.measure must use a kernel certifying union "
            f"measure; got {type(self).__name__}"
        )

    def boundary_measure(self, state: DesignState, /) -> Array:
        raise NotImplementedError(
            "MappedMeshSupportKernel.boundary_measure must use a kernel certifying "
            f"boundary measure; got {type(self).__name__}"
        )

    def sample_interior(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: PRNGKey,
        plan: RejectionSamplingPlan | None = None,
    ) -> SamplingResult:
        raise NotImplementedError(
            "MappedMeshSupportKernel.sample_interior must use a kernel providing "
            f"interior sampling; got {type(self).__name__}"
        )

    def sample_boundary(
        self, state: DesignState, num_points: int, /, *, key: PRNGKey
    ) -> SamplingResult:
        raise NotImplementedError(
            "MappedMeshSupportKernel.sample_boundary must use a kernel providing "
            f"boundary sampling; got {type(self).__name__}"
        )

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        coordinates = eqx.error_if(
            self.coordinates.read(state),
            ~self._source_current(state),
            "Mapped boundary atlas has stale coordinate-source revision evidence.",
        )
        return eqx.tree_at(
            lambda atlas: atlas.mapping.coordinates, self.atlas, coordinates
        )


@checked
def mapped_mesh_support_geometry(
    locator: AbstractCellLocator,
    support_id: str,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> CompiledGeometry:
    """Compile whole-map support, retaining explicit bounded query failures."""
    locator = locator.canonical_locator
    if not isinstance(
        locator, (PreparedMappedCellLocator, PreparedSimplicialCellLocator)
    ):
        raise TypeError(
            "mapped_support.locator must be a prepared FE locator; got another locator kind"
        )
    with _owned_mapped_support_preparation(
        locator.cell_map,
        locator.coordinates,
        budget=budget,
        include_enclosure=not isinstance(locator, PreparedMappedCellLocator),
    ) as budget:
        if not isinstance(locator, PreparedMappedCellLocator):
            locator = PreparedMappedCellLocator(
                locator.cell_map, locator.coordinates, locator.policy, budget=budget
            )
        return MappedMeshSupportSource(locator, str(support_id)).compile(
            coordinate_unit=METER, budget=budget
        )


@checked
def mapped_mesh_support_query(
    geometry: CompiledGeometry, points: Array, /
) -> MappedMeshSupportQueryResult:
    """Return membership and status rather than conflating failed inverse/outside."""
    match geometry.kernel:
        case MappedMeshSupportKernel() as kernel:
            return kernel.query(geometry.state, points)
        case _:
            raise TypeError(
                "mapped_support.geometry.kernel must represent whole mapped-cell support; got another kernel"
            )


@checked
def simplicial_mesh_support_geometry(
    locator: AbstractCellLocator,
    support_id: str,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> CompiledGeometry:
    """Retain affine distance support or compile the whole mapped domain."""

    locator = locator.canonical_locator
    if isinstance(locator, PreparedPolyhedralCellLocator):
        return locator.support_geometry(support_id)
    cell_map = locator.cell_map
    if not isinstance(cell_map, PreparedFiniteElementCellMap):
        raise TypeError(
            "mesh_support.locator must retain a prepared FE or polyhedral map; got another map kind"
        )
    element = cell_map.coordinate_element
    if (
        not isinstance(element, FiniteElementSpec)
        or element.degree != 1
        or (
            element.cell_kind
            not in ("interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron")
            and not element.cell_kind.startswith(("simplex:", "tensor:"))
        )
        or cell_map.ambient_dimension != cell_map.reference_dimension
    ):
        return mapped_mesh_support_geometry(locator, support_id, budget=budget)
    coordinates = np.asarray(locator.coordinates, dtype=np.float64)
    cells = np.asarray(cell_map.coordinate_dofs, dtype=np.int64)
    feature_id = f"finite-element-support:{support_id}"
    match cell_map.coordinate_element.cell_kind:
        case "triangle":
            source = imported_geometry_simplicial__io.planar_region_from_triangles(
                coordinates, cells, recenter=False, feature_id=feature_id
            )
        case "tetrahedron":
            faces = np.concatenate(
                (
                    cells[:, [1, 2, 3]],
                    cells[:, [0, 3, 2]],
                    cells[:, [0, 1, 3]],
                    cells[:, [0, 2, 1]],
                )
            )
            keys = np.sort(faces, axis=1)
            _, inverse, counts = np.unique(
                keys, axis=0, return_inverse=True, return_counts=True
            )
            boundary = faces[counts[inverse.reshape((-1,))] == 1]
            used, compact = np.unique(boundary, return_inverse=True)
            source = imported_geometry_simplicial.MeshRegion(
                coordinates[used],
                compact.reshape(boundary.shape).astype(np.int32),
                feature_id=feature_id,
            )
        case "quadrilateral" | "hexahedron":
            return _tensor_mesh_support_geometry(
                coordinates,
                cells,
                cell_map.coordinate_element.cell_kind,
                feature_id,
                locator,
                budget=budget,
            )
        case "interval":
            with _owned_affine_support_preparation(
                cell_map, coordinates, budget=budget
            ) as budget:
                return imported_geometry_simplicial__support_region.compile_simplicial_support(
                    locator, support_id, budget=budget
                )
        case kind:
            if kind.startswith("simplex:"):
                with _owned_affine_support_preparation(
                    cell_map, coordinates, budget=budget
                ) as budget:
                    return imported_geometry_simplicial__support_region.compile_simplicial_support(
                        locator, support_id, budget=budget
                    )
            if kind.startswith("tensor:"):
                return _tensor_mesh_support_geometry(
                    coordinates, cells, kind, feature_id, locator, budget=budget
                )
            raise ValueError(
                f"mesh_support.cell_kind must have a derived support or explicit support_geometry; got {kind!r}"
            )
    return source.compile(coordinate_unit=METER)


def _tensor_mesh_support_geometry(
    coordinates: np.ndarray,
    cells: np.ndarray,
    cell_kind: str,
    feature_id: str,
    locator: AbstractCellLocator,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> CompiledGeometry:

    reference = reference_cell_topology(cell_kind)
    if reference.dimension == 1:
        if not isinstance(locator.cell_map, PreparedFiniteElementCellMap):
            raise TypeError(
                "tensor_support.locator.cell_map must be a prepared FE coordinate map; got another map kind"
            )
        with _owned_affine_support_preparation(
            locator.cell_map, coordinates, budget=budget
        ) as budget:
            return (
                imported_geometry_simplicial__support_region.compile_simplicial_support(
                    locator, feature_id, budget=budget
                )
            )
    if reference.dimension == 2:
        triangles = cells[:, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)]
        return imported_geometry_simplicial__io.planar_region_from_triangles(
            coordinates,
            triangles.reshape((-1, 3)),
            recenter=False,
            feature_id=feature_id,
        ).compile(coordinate_unit=METER)
    if reference.dimension != 3:
        return mapped_mesh_support_geometry(locator, feature_id, budget=budget)
    faces = cells[:, np.asarray(reference.entities[2], dtype=np.int32)]
    keys = np.sort(faces.reshape((-1, 4)), axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError(
            "tensor_support.faces must be manifold; got more than two face occurrences"
        )
    exterior = counts[inverse] == 1
    boundary = faces.reshape((-1, 4))[exterior].copy()
    parent = np.repeat(np.arange(cells.shape[0]), faces.shape[1])[exterior]
    points = coordinates[boundary]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    scale = np.max(np.linalg.norm(points - points[:, :1], axis=2), axis=1)
    if np.any(norm <= np.finfo(np.float64).eps * np.maximum(scale**2, 1.0)):
        raise ValueError(
            "tensor_support.faces must be nondegenerate; got a degenerate boundary face"
        )
    residual = np.abs(np.sum((points[:, 3] - points[:, 0]) * normals, axis=1)) / norm
    if np.any(residual > 1e-10 * np.maximum(scale, 1.0)):
        return mapped_mesh_support_geometry(locator, feature_id, budget=budget)
    centers = np.mean(coordinates[cells[parent]], axis=1)
    inward = np.sum(normals * (np.mean(points, axis=1) - centers), axis=1) < 0.0
    boundary[inward] = boundary[inward][:, (0, 3, 2, 1)]
    triangles = boundary[:, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)].reshape(
        (-1, 3)
    )
    used, compact = np.unique(triangles, return_inverse=True)
    return imported_geometry_simplicial.MeshRegion(
        coordinates[used],
        compact.reshape(triangles.shape).astype(np.int32),
        feature_id=feature_id,
    ).compile(coordinate_unit=METER)


class _SupportBoundary(StrictModule):
    """Represented affine boundary and direct identity comparison evidence."""

    vertices: np.ndarray
    facets: np.ndarray
    matched: bool = eqx.field(static=True)


def _boundary_keys(points: np.ndarray, /) -> list[tuple[tuple[float, ...], ...]]:
    return sorted(
        tuple(sorted(tuple(float(value) for value in vertex) for vertex in face))
        for face in points
    )


def _simplicial_support_facets(
    kernel: imported_geometry_simplicial__support_region.SimplicialSupportKernel,
    /,
) -> np.ndarray:
    workset = kernel.projections[-1]
    origins = np.asarray(workset.origins)
    return np.concatenate(
        (
            origins[:, None, :],
            origins[:, None, :] + np.swapaxes(np.asarray(workset.edges), -1, -2),
        ),
        axis=1,
    )


def _prepare_affine_support_boundary(
    canonical: CompiledGeometry, supplied: CompiledGeometry, /
) -> _SupportBoundary:
    match canonical.kernel:
        case (
            imported_geometry_simplicial__support_region.SimplicialSupportKernel() as kernel
        ):
            facets = _simplicial_support_facets(kernel)
            vertices = facets.reshape((-1, kernel.dimension))
            match supplied.kernel:
                case (
                    imported_geometry_simplicial__support_region.SimplicialSupportKernel() as other
                ):
                    matched = _boundary_keys(facets) == _boundary_keys(
                        _simplicial_support_facets(other)
                    )
                case _:
                    matched = False
        case imported_geometry_simplicial__regions.MeshRegionKernel() as kernel:
            vertices = np.asarray(kernel.vertices.read(canonical.state), dtype=np.float64)
            facets = vertices[np.asarray(kernel.faces)]
            matched = False
        case imported_geometry_simplicial__regions.PlanarMeshRegionKernel() as kernel:
            vertices = np.asarray(kernel.vertices.read(canonical.state), dtype=np.float64)
            facets = vertices[np.asarray(kernel.edges)]
            matched = False
        case _:
            raise ValueError(
                "mesh_support.support_geometry must retain whole represented-domain evidence; got unsupported boundary evidence"
            )
    return _SupportBoundary(vertices, facets, matched)


def _verify_mapped_support_binding(
    geometry: CompiledGeometry,
    kernel: MappedMeshSupportKernel,
    locator: AbstractCellLocator,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> None:
    match locator:
        case PreparedMappedCellLocator():
            mapped = locator
        case PreparedSimplicialCellLocator():
            mapped = PreparedMappedCellLocator(
                locator.cell_map, locator.coordinates, locator.policy, budget=budget
            )
        case _:
            raise TypeError(
                "mapped_support.locator must be a prepared FE locator; got another locator kind"
            )
    if kernel.locator.source_binding_id != mapped.source_binding_id:
        raise ValueError(
            "mapped_support.support_geometry must bind the coordinate-map source revision; got a different source binding"
        )
    if not bool(np.asarray(kernel._source_current(geometry.state))):
        raise ValueError(
            "mapped_support.support_geometry must retain current whole-map bounds; got stale source bounds"
        )


@checked
def verify_mesh_support_geometry(
    geometry: CompiledGeometry,
    locator: AbstractCellLocator,
    /,
    *,
    tolerance: float,
    budget: CoordinateEnclosureBudget | None = None,
) -> None:
    """Require source binding or whole affine-boundary coverage evidence."""
    locator = locator.canonical_locator
    limit = float(tolerance)
    if not isfinite(limit) or limit < 0.0:
        raise ValueError(
            "tensor_box_support.support_tolerance must be finite and nonnegative; got an invalid tolerance"
        )
    match geometry.kernel:
        case MappedMeshSupportKernel() as kernel:
            _verify_mapped_support_binding(geometry, kernel, locator, budget=budget)
            return
        case _:
            pass
    canonical = simplicial_mesh_support_geometry(
        locator, "support-verification", budget=budget
    )
    if isinstance(canonical.kernel, MappedMeshSupportKernel):
        raise ValueError(
            "mesh_support.support_geometry must cover the whole mapped source; got unmatched whole-map source binding"
        )
    boundary = _prepare_affine_support_boundary(canonical, geometry)
    if boundary.matched:
        return
    vertices, facets = boundary.vertices, boundary.facets
    match geometry.kernel:
        case imported_geometry_analytic__primitives.OrthotopeKernel():
            box = np.asarray(geometry.bounds, dtype=np.float64)
            scale = max(1.0, float(np.max(np.abs(box))))
            same_bounds = (
                np.max(
                    np.abs(np.stack((vertices.min(axis=0), vertices.max(axis=0))) - box)
                )
                <= limit * scale
            )
            on_planes = np.any(
                np.all(
                    np.abs(facets[:, :, None, :] - box[None, None, :, :])
                    <= limit * scale,
                    axis=1,
                ),
                axis=(1, 2),
            )
            # Closed boundary coverage includes every extremal plane and hole.
            if same_bounds and np.all(on_planes):
                return
        case imported_geometry_simplicial__regions.MeshRegionKernel() as kernel if type(
            kernel
        ) is type(canonical.kernel):
            supplied_vertices = np.asarray(
                kernel.vertices.read(geometry.state), dtype=np.float64
            )
            if _boundary_keys(facets) == _boundary_keys(
                supplied_vertices[np.asarray(kernel.faces)]
            ):
                return
        case imported_geometry_simplicial__regions.PlanarMeshRegionKernel() as kernel if (
            type(kernel) is type(canonical.kernel)
        ):
            supplied_vertices = np.asarray(
                kernel.vertices.read(geometry.state), dtype=np.float64
            )
            if _boundary_keys(facets) == _boundary_keys(
                supplied_vertices[np.asarray(kernel.edges)]
            ):
                return
        case _:
            pass
    raise ValueError(
        "mesh_support.support_geometry must cover the represented mesh boundary; got unmatched whole-domain boundary evidence"
    )


@checked
def tensor_box_support_geometry(
    lower: ArrayLike,
    upper: ArrayLike,
    support_geometry: imported_geometry.CompiledGeometry | None,
    /,
    *,
    feature_id: str,
    tolerance: float,
) -> imported_geometry.CompiledGeometry:
    """Return the compiled support box, or verify an explicit box geometry.

    `support_geometry=None` compiles an `Orthotope` spanning `[lower, upper]`.
    An explicit region is admitted only when its bounds equal the box and its
    interior measure equals the box measure: a region inside the box's bounding
    box with the box's measure is the box.
    """

    lower_ = np.asarray(lower, dtype=np.float64)
    upper_ = np.asarray(upper, dtype=np.float64)
    if (
        lower_.ndim != 1
        or upper_.shape != lower_.shape
        or not np.all(np.isfinite(lower_) & np.isfinite(upper_))
        or np.any(upper_ <= lower_)
    ):
        raise ValueError(
            "tensor_box_support.bounds must contain finite ordered lower and upper corners; got invalid box bounds"
        )
    limit = float(tolerance)
    if not isfinite(limit) or limit < 0.0:
        raise ValueError(
            "tensor_box_support.support_tolerance must be finite and nonnegative; got an invalid tolerance"
        )
    if support_geometry is None:
        return imported_geometry.Orthotope(
            0.5 * (lower_ + upper_), upper_ - lower_, feature_id=feature_id
        ).compile(coordinate_unit=METER)
    if support_geometry.kind is not imported_geometry.GeometryKind.REGION:
        raise ValueError(
            "tensor_box_support.support_geometry must be a region geometry; got another geometry kind"
        )
    if support_geometry.ambient_dimension != lower_.size:
        raise ValueError(
            "tensor_box_support.support_geometry.dimension must equal the box dimension; got a different ambient dimension"
        )
    if not support_geometry.has_capability(
        imported_geometry.GeometryCapability.INTERIOR_MEASURE
    ):
        raise ValueError(
            "tensor_box_support.support_geometry.interior_measure must evidence the tensor box; got no interior measure"
        )
    box = np.stack((lower_, upper_))
    bounds = np.asarray(support_geometry.bounds, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(box))))
    if bounds.shape != box.shape or np.max(np.abs(bounds - box)) > limit * scale:
        raise ValueError(
            "tensor_box_support.support_geometry.bounds must match the reconstruction box; got different bounds"
        )
    box_measure = float(np.prod(upper_ - lower_))
    measure = float(np.asarray(support_geometry.measure))
    if abs(measure - box_measure) > limit * max(1.0, box_measure):
        raise ValueError(
            "tensor_box_support.support_geometry.interior_measure must match the reconstruction box; got a different measure"
        )
    return support_geometry


__all__ = [
    "cell_location_status",
    "MappedMeshSupportQueryResult",
    "MappedMeshSupportSource",
    "MappedSupportQueryStatus",
    "mapped_mesh_support_geometry",
    "mapped_mesh_support_query",
    "simplicial_mesh_support_geometry",
    "tensor_box_support_geometry",
    "verify_mesh_support_geometry",
]
