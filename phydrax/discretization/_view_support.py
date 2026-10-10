#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Support geometry and location status shared by discrete field views.

Structured reconstructions cover a tensor box; simplicial boundaries retain
their affine distance provider; mapped cells retain the entire coordinate
source. Explicit support must carry whole-domain binding or boundary evidence.
"""

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._strict import StrictModule
from ..geometry._atlas import BoundaryAtlas, BoundaryMap
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
)
from ..geometry.design._schema import (
    _ParameterCollector,
    DesignState,
    ParameterBinding,
    ParameterId,
)
from ..typing import PRNGKey
from ._cell_complex import (
    IntervalConnectivity,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ._cell_mesh import SimplicialConnectivity
from ._coordinate_enclosure import (
    coordinate_expressions,
    Expression,
    restrict_chart_expressions,
)
from ._hexahedral import HexahedralConnectivity
from ._mapped_locator import PreparedMappedCellLocator
from ._polyhedral_locator import PreparedPolyhedralCellLocator
from ._reference_cell import reference_cell_topology
from ._simplicial_locator import (
    _cell_map_vertices,
    AbstractCellLocator,
    CellLocationResult,
    CellLocationStatus,
    PreparedSimplicialCellLocator,
)
from ._views import FieldQueryStatus
from .fem._cell_map import PreparedFiniteElementCellMap
from .fem._reference import FiniteElementSpec
from .fem._reference_operator import reference_facet_embedding


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


class _MappedCellBoundaryMap(BoundaryMap):
    """Canonical reference facets composed with the complete coordinate source."""

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
    """Canonical independent directions in the given global facet-vertex order."""
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
        raise ValueError("Reference facet corners must have codimension one.")
    return np.stack(directions, axis=1) if directions else np.empty((dimension, 0))


def _mapped_mesh_boundary_atlas(locator: PreparedMappedCellLocator, /) -> BoundaryAtlas:
    """Compile oriented exterior reference facets with canonical mesh identities."""
    cell_map = locator.cell_map
    mesh = cell_map.mesh
    kind = cell_map.coordinate_element.cell_kind
    topology = reference_cell_topology(kind)
    facets = topology.entities[topology.dimension - 1]
    cells = _cell_map_vertices(cell_map)
    connectivity = mesh.connectivity
    if isinstance(connectivity, SimplicialConnectivity):
        global_facets = tuple(
            tuple(int(value) for value in row)
            for row in np.asarray(connectivity.entities[topology.dimension - 1])
        )
    elif topology.dimension == 1 and isinstance(connectivity, IntervalConnectivity):
        global_facets = tuple((index,) for index in range(mesh.coordinates.shape[0]))
    elif topology.dimension == 2 and isinstance(connectivity, PolygonalConnectivity):
        global_facets = tuple(
            tuple(int(value) for value in row) for row in np.asarray(connectivity.edges)
        )
    elif topology.dimension == 3 and isinstance(
        connectivity, (TetrahedralConnectivity, HexahedralConnectivity)
    ):
        global_facets = tuple(
            tuple(int(value) for value in row) for row in np.asarray(connectivity.faces)
        )
    elif topology.dimension == 3 and isinstance(connectivity, PolyhedralConnectivity):
        offsets = np.asarray(connectivity.face_vertex_offsets)
        values = np.asarray(connectivity.face_vertex_values)
        global_facets = tuple(
            tuple(int(value) for value in values[start:stop])
            for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
        )
    else:
        raise TypeError(
            "Mapped boundary facets require canonical connectivity matching the reference topology."
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

    # Interior topology may only erase a face when its full coordinate-source
    # traces agree. Matching corners or floating samples are not sufficient.
    source_polynomials: dict[int, tuple[Expression, ...] | None] = {}
    routes = np.asarray(cell_map.coordinate_dofs)
    coordinates = cell_map.source_coordinates(locator.coordinates)

    def trace(cell: int, slot: int) -> tuple[Expression, ...]:
        if cell not in source_polynomials:
            source_polynomials[cell] = coordinate_expressions(
                cell_map.coordinate_element,
                tuple(coordinates[index] for index in routes[cell]),
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
                "Mapped support traces require their actual owned source expressions."
            )
        target_kind = (
            "interval"
            if axes.shape[1] == 1
            else "triangle"
            if len(corners) == 3
            else "quadrilateral"
        )
        return restrict_chart_expressions(polynomials, kind, target_kind, origin, axes)

    for slot, entries in occurrences.items():
        if len(entries) == 2 and trace(entries[0][0], slot) != trace(entries[1][0], slot):
            raise ValueError(
                "Mapped support topology has nonmatching full coordinate-source facet traces."
            )
        if len(entries) > 2:
            raise ValueError(
                "Mapped support boundary requires manifold source facet topology."
            )
    facet_ids = np.asarray(mesh.entity_set(topology.dimension - 1).entity_ids)
    exterior = sorted(
        (slot, entries[0][0], entries[0][1])
        for slot, entries in occurrences.items()
        if len(entries) == 1
    )
    if not exterior:
        raise ValueError(
            "Mapped support requires a non-empty represented exterior boundary."
        )
    ids = np.asarray([facet_ids[slot] for slot, _, _ in exterior], dtype=np.int64)
    if np.any(ids > np.iinfo(np.int32).max):
        raise ValueError(
            "Mapped boundary atlas source facet IDs exceed represented int32 capacity."
        )
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
        embedded, _ = reference_facet_embedding(kind, index, parameter)
        facet_axes[index] = np.asarray(embedded[1:] - embedded[:1]).T
        raw_normal = np.asarray(
            tuple(
                (-1.0) ** axis * np.linalg.det(np.delete(facet_axes[index], axis, axis=0))
                for axis in range(topology.dimension)
            )
        )
        outward = corners.mean(axis=0) - np.asarray(topology.vertices).mean(axis=0)
        signs[index] = 1.0 if raw_normal @ outward > 0.0 else -1.0
        # Unnormalized dyadic reference planes preserve their actual zero set;
        # their margin is not a signed physical or reference Euclidean distance.
        facet_normals[index] = signs[index] * raw_normal
        facet_offsets[index] = facet_normals[index] @ corners[0]
    mapping = _MappedCellBoundaryMap(
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
    """Union of actual prepared coordinate-map images, not corner polytopes."""

    locator: PreparedMappedCellLocator
    support_id: str = eqx.field(static=True)

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        coordinates = context.bind(
            ParameterId(f"mapped-cell-support:{self.support_id}", "coordinates"),
            self.locator.coordinates,
            role="position",
            trainable=False,
        )
        return _MappedMeshSupportKernel(
            self.locator, coordinates, _mapped_mesh_boundary_atlas(self.locator)
        )


class _MappedMeshSupportKernel(GeometryKernel):
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
            raise ValueError("Mapped support points have incompatible ambient dimension.")
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
        assert isinstance(mapping, _MappedCellBoundaryMap)
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
            "Mapped membership support does not provide physical boundary normals."
        )

    def measure(self, state: DesignState, /) -> Array:
        raise NotImplementedError(
            "Mapped membership support does not certify union measure."
        )

    def boundary_measure(self, state: DesignState, /) -> Array:
        raise NotImplementedError(
            "Mapped membership support does not certify boundary measure."
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
            "Mapped membership support does not provide interior sampling."
        )

    def sample_boundary(
        self, state: DesignState, num_points: int, /, *, key: PRNGKey
    ) -> SamplingResult:
        raise NotImplementedError(
            "Mapped membership support does not provide boundary sampling."
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


def mapped_mesh_support_geometry(
    locator: AbstractCellLocator, support_id: str, /
) -> CompiledGeometry:
    """Compile whole-map support, retaining explicit bounded query failures."""
    locator = locator.canonical_locator
    if not isinstance(locator, PreparedMappedCellLocator):
        if not isinstance(locator, PreparedSimplicialCellLocator):
            raise TypeError("Mapped support requires a canonical prepared FE locator.")
        locator = PreparedMappedCellLocator(
            locator.cell_map, locator.coordinates, locator.policy
        )
    return MappedMeshSupportSource(locator, str(support_id)).compile()


def mapped_mesh_support_query(
    geometry: CompiledGeometry, points: Array, /
) -> MappedMeshSupportQueryResult:
    """Return membership and status rather than conflating failed inverse/outside."""
    if not isinstance(geometry.kernel, _MappedMeshSupportKernel):
        raise TypeError("Geometry is not whole mapped-cell support.")
    return geometry.kernel.query(geometry.state, points)


def simplicial_mesh_support_geometry(
    locator: AbstractCellLocator, support_id: str, /
) -> CompiledGeometry:
    """Retain affine distance support or compile the actual whole mapped domain."""
    from ..geometry.simplicial import MeshRegion
    from ..geometry.simplicial._io import planar_region_from_triangles

    locator = locator.canonical_locator
    if isinstance(locator, PreparedPolyhedralCellLocator):
        return locator.support_geometry(support_id)
    cell_map = locator.cell_map
    if not isinstance(cell_map, PreparedFiniteElementCellMap):
        raise TypeError(
            "Mesh support requires a canonical prepared FE or polyhedral locator."
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
        return mapped_mesh_support_geometry(locator, support_id)
    coordinates = np.asarray(locator.coordinates, dtype=np.float64)
    cells = np.asarray(cell_map.coordinate_dofs, dtype=np.int64)
    feature_id = f"finite-element-support:{support_id}"
    match cell_map.coordinate_element.cell_kind:
        case "triangle":
            source = planar_region_from_triangles(
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
            source = MeshRegion(
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
            )
        case "interval":
            from ..geometry.simplicial._support_region import compile_simplicial_support

            return compile_simplicial_support(locator, support_id)
        case kind:
            if kind.startswith("simplex:"):
                from ..geometry.simplicial._support_region import (
                    compile_simplicial_support,
                )

                return compile_simplicial_support(locator, support_id)
            if kind.startswith("tensor:"):
                return _tensor_mesh_support_geometry(
                    coordinates, cells, kind, feature_id, locator
                )
            raise ValueError(
                f"The support of {kind!r} FE cells is not derived; pass an explicit support_geometry."
            )
    return source.compile()


def _tensor_mesh_support_geometry(
    coordinates: np.ndarray,
    cells: np.ndarray,
    cell_kind: str,
    feature_id: str,
    locator: AbstractCellLocator,
    /,
) -> CompiledGeometry:
    from ..geometry.simplicial import MeshRegion
    from ..geometry.simplicial._io import planar_region_from_triangles

    reference = reference_cell_topology(cell_kind)
    if reference.dimension == 1:
        from ..geometry.simplicial._support_region import compile_simplicial_support

        return compile_simplicial_support(locator, feature_id)
    if reference.dimension == 2:
        triangles = cells[:, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)]
        return planar_region_from_triangles(
            coordinates,
            triangles.reshape((-1, 3)),
            recenter=False,
            feature_id=feature_id,
        ).compile()
    if reference.dimension != 3:
        return mapped_mesh_support_geometry(locator, feature_id)
    faces = cells[:, np.asarray(reference.entities[2], dtype=np.int32)]
    keys = np.sort(faces.reshape((-1, 4)), axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Tensor support requires manifold faces.")
    exterior = counts[inverse] == 1
    boundary = faces.reshape((-1, 4))[exterior].copy()
    parent = np.repeat(np.arange(cells.shape[0]), faces.shape[1])[exterior]
    points = coordinates[boundary]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    scale = np.max(np.linalg.norm(points - points[:, :1], axis=2), axis=1)
    if np.any(norm <= np.finfo(np.float64).eps * np.maximum(scale**2, 1.0)):
        raise ValueError("Tensor support boundary faces must be nondegenerate.")
    residual = np.abs(np.sum((points[:, 3] - points[:, 0]) * normals, axis=1)) / norm
    if np.any(residual > 1e-10 * np.maximum(scale, 1.0)):
        return mapped_mesh_support_geometry(locator, feature_id)
    centers = np.mean(coordinates[cells[parent]], axis=1)
    inward = np.sum(normals * (np.mean(points, axis=1) - centers), axis=1) < 0.0
    boundary[inward] = boundary[inward][:, (0, 3, 2, 1)]
    triangles = boundary[:, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)].reshape(
        (-1, 3)
    )
    used, compact = np.unique(triangles, return_inverse=True)
    return MeshRegion(
        coordinates[used],
        compact.reshape(triangles.shape).astype(np.int32),
        feature_id=feature_id,
    ).compile()


def verify_mesh_support_geometry(
    geometry: CompiledGeometry, locator: AbstractCellLocator, /, *, tolerance: float
) -> None:
    """Require source binding or exact whole affine-boundary coverage evidence."""
    from ..geometry.analytic._primitives import _OrthotopeKernel
    from ..geometry.simplicial._regions import _MeshRegionKernel, _PlanarMeshRegionKernel
    from ..geometry.simplicial._support_region import _SimplicialSupportKernel

    locator = locator.canonical_locator
    limit = float(tolerance)
    if not isfinite(limit) or limit < 0.0:
        raise ValueError("support_tolerance must be finite and non-negative.")
    if not isinstance(geometry, CompiledGeometry):
        raise TypeError("support_geometry must be CompiledGeometry.")
    if isinstance(geometry.kernel, _MappedMeshSupportKernel):
        if isinstance(locator, PreparedMappedCellLocator):
            mapped = locator
        else:
            if not isinstance(locator, PreparedSimplicialCellLocator):
                raise TypeError(
                    "Mapped support verification requires a canonical prepared FE locator."
                )
            mapped = PreparedMappedCellLocator(
                locator.cell_map, locator.coordinates, locator.policy
            )
        if geometry.kernel.locator.source_binding_id != mapped.source_binding_id:
            raise ValueError(
                "The support geometry does not cover this coordinate-map source revision."
            )
        if not bool(np.asarray(geometry.kernel._source_current(geometry.state))):
            raise ValueError("The support geometry has stale whole-map source bounds.")
        return
    canonical = simplicial_mesh_support_geometry(locator, "support-verification")
    if isinstance(canonical.kernel, _MappedMeshSupportKernel):
        raise ValueError(
            "Explicit support_geometry does not cover the whole mapped source; "
            "use matching mapped source binding, not corner inclusion or sampled volume."
        )
    kernel = canonical.kernel
    if isinstance(kernel, _SimplicialSupportKernel):
        workset = kernel.projections[-1]
        origins = np.asarray(workset.origins)
        facets = np.concatenate(
            (
                origins[:, None, :],
                origins[:, None, :] + np.swapaxes(np.asarray(workset.edges), -1, -2),
            ),
            axis=1,
        )
        vertices = facets.reshape((-1, kernel.dimension))
        if isinstance(geometry.kernel, _SimplicialSupportKernel):
            supplied = geometry.kernel.projections[-1]
            supplied_origins = np.asarray(supplied.origins)
            supplied_facets = np.concatenate(
                (
                    supplied_origins[:, None, :],
                    supplied_origins[:, None, :]
                    + np.swapaxes(np.asarray(supplied.edges), -1, -2),
                ),
                axis=1,
            )

            def keys(points: np.ndarray) -> list[tuple[tuple[float, ...], ...]]:
                return sorted(
                    tuple(
                        sorted(tuple(float(value) for value in vertex) for vertex in face)
                    )
                    for face in points
                )

            if keys(facets) == keys(supplied_facets):
                return
    elif isinstance(kernel, (_MeshRegionKernel, _PlanarMeshRegionKernel)):
        vertices = np.asarray(kernel.vertices.read(canonical.state), dtype=np.float64)
        routes = np.asarray(
            kernel.faces if isinstance(kernel, _MeshRegionKernel) else kernel.edges
        )
        facets = vertices[routes]
    else:
        raise ValueError(
            "Explicit support_geometry needs whole represented-domain support evidence."
        )
    if isinstance(geometry.kernel, _OrthotopeKernel):
        box = np.asarray(geometry.bounds, dtype=np.float64)
        scale = max(1.0, float(np.max(np.abs(box))))
        same_bounds = (
            np.max(np.abs(np.stack((vertices.min(axis=0), vertices.max(axis=0))) - box))
            <= limit * scale
        )
        on_planes = np.any(
            np.all(
                np.abs(facets[:, :, None, :] - box[None, None, :, :]) <= limit * scale,
                axis=1,
            ),
            axis=(1, 2),
        )
        # The canonical closed boundary lies wholly on box facets and reaches
        # every extremal plane. No interior hole/cut boundary is discarded.
        if same_bounds and np.all(on_planes):
            return
    elif isinstance(
        geometry.kernel, (_MeshRegionKernel, _PlanarMeshRegionKernel)
    ) and type(geometry.kernel) is type(kernel):
        supplied_vertices = np.asarray(
            geometry.kernel.vertices.read(geometry.state), dtype=np.float64
        )
        supplied_routes = np.asarray(
            geometry.kernel.faces
            if isinstance(geometry.kernel, _MeshRegionKernel)
            else geometry.kernel.edges
        )

        def boundary_keys(points: np.ndarray) -> list[tuple[tuple[float, ...], ...]]:
            return sorted(
                tuple(sorted(tuple(float(value) for value in vertex) for vertex in face))
                for face in points
            )

        if boundary_keys(facets) == boundary_keys(supplied_vertices[supplied_routes]):
            return
    raise ValueError(
        "The explicit support geometry does not cover the exact mesh boundary; "
        "corner inclusion and equal sampled measure are not whole-domain evidence."
    )


def tensor_box_support_geometry(
    lower: np.ndarray,
    upper: np.ndarray,
    support_geometry: CompiledGeometry | None,
    /,
    *,
    feature_id: str,
    tolerance: float,
) -> CompiledGeometry:
    """Return the compiled support box, or verify an explicit box geometry.

    `support_geometry=None` compiles an `Orthotope` spanning `[lower, upper]`.
    An explicit region is admitted only when its bounds equal the box and its
    interior measure equals the box measure: a region inside the box's bounding
    box with the box's measure is the box.
    """
    from ..geometry import CompiledGeometry, GeometryCapability, GeometryKind, Orthotope

    lower_ = np.asarray(lower, dtype=np.float64)
    upper_ = np.asarray(upper, dtype=np.float64)
    if (
        lower_.ndim != 1
        or upper_.shape != lower_.shape
        or not np.all(np.isfinite(lower_) & np.isfinite(upper_))
        or np.any(upper_ <= lower_)
    ):
        raise ValueError("A support box needs finite, ordered lower and upper corners.")
    limit = float(tolerance)
    if not isfinite(limit) or limit < 0.0:
        raise ValueError("support_tolerance must be finite and non-negative.")
    if support_geometry is None:
        return Orthotope(
            0.5 * (lower_ + upper_), upper_ - lower_, feature_id=feature_id
        ).compile()
    if not isinstance(support_geometry, CompiledGeometry):
        raise TypeError("support_geometry must be a CompiledGeometry or None.")
    if support_geometry.kind is not GeometryKind.REGION:
        raise ValueError("support_geometry must be a region geometry.")
    if support_geometry.ambient_dimension != lower_.size:
        raise ValueError("support_geometry dimension must equal the box dimension.")
    if not support_geometry.has_capability(GeometryCapability.INTERIOR_MEASURE):
        raise ValueError(
            "An explicit support_geometry needs an interior measure to evidence "
            "that it is the reconstruction's tensor box."
        )
    box = np.stack((lower_, upper_))
    bounds = np.asarray(support_geometry.bounds, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(box))))
    if bounds.shape != box.shape or np.max(np.abs(bounds - box)) > limit * scale:
        raise ValueError("support_geometry bounds differ from the reconstruction box.")
    box_measure = float(np.prod(upper_ - lower_))
    measure = float(np.asarray(support_geometry.measure))
    if abs(measure - box_measure) > limit * max(1.0, box_measure):
        raise ValueError(
            "support_geometry measure differs from the reconstruction box; the "
            "geometry is not the tensor box the reconstruction covers."
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
